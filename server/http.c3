module server::http;

import std::core::mem;
import std::io;
import std::net;

faultdef BAD_REQUEST, REQUEST_TOO_LARGE, READ_ERROR;

const usz MAX_HEADER_BYTES = 16 * 1024;          // 16 KiB headers cap
const usz MAX_BODY_BYTES   = 64 * 1024 * 1024;   // 64 MiB body cap (base64 PNG fits)

struct HttpRequest
{
    String method;
    String path;
    char[] body;
    char[] _buf;   // owns the underlying header bytes; method/path slice into it
}

fn void HttpRequest.free(&self)
{
    if (self._buf.len > 0) mem::free(self._buf);
    if (self.body.len > 0) mem::free(self.body);
}

// Read one full HTTP/1.1 request from the socket. Caller must call .free()
// on the returned request.
fn HttpRequest? read_request(net::Socket* sock)
{
    // Grow a header buffer until we see "\r\n\r\n".
    usz cap = 4096;
    char[] hbuf = mem::new_array(char, cap);
    usz n = 0;
    usz hdr_end = 0;
    bool found = false;
    while (!found)
    {
        if (n == cap)
        {
            if (cap >= MAX_HEADER_BYTES) { mem::free(hbuf); return REQUEST_TOO_LARGE~; }
            usz new_cap = cap * 2;
            if (new_cap > MAX_HEADER_BYTES) new_cap = MAX_HEADER_BYTES;
            char[] grown = mem::new_array(char, new_cap);
            for (usz i = 0; i < n; i++) grown[i] = hbuf[i];
            mem::free(hbuf);
            hbuf = grown;
            cap = new_cap;
        }
        usz r;
        if (try got = sock.read(hbuf[n..cap - 1]))
        {
            r = got;
        }
        else
        {
            mem::free(hbuf);
            return READ_ERROR~;
        }
        if (r == 0) { mem::free(hbuf); return READ_ERROR~; }
        n += r;
        // Look for \r\n\r\n in the new bytes (with overlap of 3).
        usz scan_start = n - r;
        if (scan_start >= 3) scan_start -= 3;
        for (usz i = scan_start; i + 4 <= n; i++)
        {
            if (hbuf[i] == '\r' && hbuf[i + 1] == '\n' && hbuf[i + 2] == '\r' && hbuf[i + 3] == '\n')
            {
                hdr_end = i;
                found = true;
                break;
            }
        }
    }

    // Parse request line: "METHOD SP PATH SP HTTP/x.y\r\n"
    String headers_view = ((String)hbuf[..hdr_end - 1]);

    usz sp1 = 0;
    bool sp1_found = false;
    for (usz i = 0; i < headers_view.len; i++)
    {
        if (headers_view[i] == ' ') { sp1 = i; sp1_found = true; break; }
    }
    if (!sp1_found) { mem::free(hbuf); return BAD_REQUEST~; }

    usz sp2 = 0;
    bool sp2_found = false;
    for (usz i = sp1 + 1; i < headers_view.len; i++)
    {
        if (headers_view[i] == ' ') { sp2 = i; sp2_found = true; break; }
    }
    if (!sp2_found) { mem::free(hbuf); return BAD_REQUEST~; }

    usz line_end = 0;
    bool line_end_found = false;
    for (usz i = sp2 + 1; i + 1 < headers_view.len; i++)
    {
        if (headers_view[i] == '\r' && headers_view[i + 1] == '\n')
        {
            line_end = i;
            line_end_found = true;
            break;
        }
    }
    if (!line_end_found) { mem::free(hbuf); return BAD_REQUEST~; }

    String method = headers_view[..sp1 - 1];
    String path   = headers_view[sp1 + 1..sp2 - 1];

    // Find Content-Length header (case-insensitive).
    usz content_length = 0;
    {
        usz cursor = line_end + 2;
        while (cursor + 2 < headers_view.len)
        {
            usz le = cursor;
            for (usz i = cursor; i + 1 < headers_view.len; i++)
            {
                if (headers_view[i] == '\r' && headers_view[i + 1] == '\n') { le = i; break; }
            }
            if (le == cursor) break;

            String hline = headers_view[cursor..le - 1];
            if (hline.len > 15)
            {
                bool match = true;
                String want = "content-length:";
                for (usz i = 0; i < 15; i++)
                {
                    char a = hline[i];
                    if (a >= 'A' && a <= 'Z') a = (char)(a + 32);
                    if (a != want[i]) { match = false; break; }
                }
                if (match)
                {
                    usz vstart = 15;
                    while (vstart < hline.len && (hline[vstart] == ' ' || hline[vstart] == '\t')) vstart++;
                    usz val = 0;
                    for (usz i = vstart; i < hline.len; i++)
                    {
                        char c = hline[i];
                        if (c < '0' || c > '9') break;
                        val = val * 10 + (usz)(c - '0');
                    }
                    content_length = val;
                }
            }
            cursor = le + 2;
        }
    }

    if (content_length > MAX_BODY_BYTES) { mem::free(hbuf); return REQUEST_TOO_LARGE~; }

    // Header buffer may already contain part of the body.
    usz body_in_hbuf = n - (hdr_end + 4);
    char[] body = (char[]){};
    if (content_length > 0)
    {
        body = mem::new_array(char, content_length);
        usz copied = body_in_hbuf < content_length ? body_in_hbuf : content_length;
        for (usz i = 0; i < copied; i++) body[i] = hbuf[hdr_end + 4 + i];
        usz remaining = content_length - copied;
        usz offset = copied;
        while (remaining > 0)
        {
            usz r;
            if (try got = sock.read(body[offset..offset + remaining - 1]))
            {
                r = got;
            }
            else
            {
                mem::free(hbuf);
                mem::free(body);
                return READ_ERROR~;
            }
            if (r == 0) { mem::free(hbuf); mem::free(body); return READ_ERROR~; }
            offset    += r;
            remaining -= r;
        }
    }

    HttpRequest result = {
        .method = method,
        .path   = path,
        .body   = body,
        ._buf   = hbuf,
    };
    return result;
}

// Send `len` bytes from `data`, looping over partial writes. Errors are
// silently swallowed — there's nothing useful to do with a half-sent reply.
fn void write_all(net::Socket* sock, char[] data)
{
    usz off = 0;
    while (off < data.len)
    {
        if (try w = sock.write(data[off..data.len - 1]))
        {
            if (w == 0) return;
            off += w;
        }
        else
        {
            return;
        }
    }
}

fn String status_text(int status)
{
    switch (status)
    {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 500: return "Internal Server Error";
        default:  return "OK";
    }
}

fn void write_response(net::Socket* sock, int status, String content_type, char[] body)
{
    char[1024] hdr_buf;
    char[] hdr = io::bprintf(&hdr_buf,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",
        status, status_text(status), content_type, body.len) ?? (char[]){};
    if (hdr.len > 0) write_all(sock, hdr);
    if (body.len > 0) write_all(sock, body);
}
