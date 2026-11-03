<*
 The creature plugin's rig fit (`llm/plugins/creature/fit.js`, RigFit) in C3:
 its forward kinematics, loss, hand-written gradients and Adam, which a JS
 interpreter runs slowly. A line-for-line port in doubles, in the same order, so the two
 agree to rounding; fit.js packs the problem (RigFit.native) and keeps the JS
 as the reference.

 `rigFitEvaluate(ints, reals, obs, params, grads, tau, limitScale)` fills
 grads and answers the terms. `rigFitSteps(ints, reals, obs, params, grads,
 adamM, adamV, lr, from, to, iterations, tau0, tau1, adamT)` runs iterations
 from..to of a fit of `iterations`, the params updated in place, and answers
 the last terms, the loss, the count, Adam's t and the limit scale.
*>
module llm::host;

import three;
import fastjs::js;
import fastjs::js::opt;
import std::math;
import std::core::mem;

extern fn double c_log1p(double x) @cname("log1p");
extern fn double c_fmod(double x, double y) @cname("fmod");
extern fn double c_atan2(double y, double x) @cname("atan2");

// fit.js's ACC_SCALE: pixels a frame squared where an acceleration's cost turns from square to linear.
const double ACC_SCALE = 3;
const int FIT_TERMS = 14;
const int FIT_WEIGHTS = 18;
// The answer: the terms in fit.js's order (NaN when absent), then the loss,
// the observed count, Adam's t and the limit scale.
const int FIT_ANSWER = FIT_TERMS + 4;
const int T_REPROJ = 0;
const int T_ACC = 1;
const int T_FLOOR = 2;
const int T_IN_PLACE = 3;
const int T_MARKS = 4;
const int T_LATERAL = 5;
const int T_HEADING = 6;
const int T_TWIST = 7;
const int T_REST = 8;
const int T_CHAIN = 9;
const int T_THETA_ACC = 10;
const int T_LIMIT = 11;
const int T_LEVEL = 12;
const int T_FOLLOW = 13;

// DEFAULT_WEIGHTS' order.
enum FitWeight : int
{
	REPROJ, ACC, THETA_ACC, TWIST, TWIST_MARKED, REST, ONE_MARKER, UNSEEN, CHAIN, CHAIN_BRANCH,
	FLOOR, IN_PLACE, MARKS, FOLLOW, LEVEL, LATERAL, HEADING, LIMIT,
}

struct FitView
{
	int p;
	int h;
	int* points;    // [P][joint, marker]
	double* x;
	double* y;
	double* w;
	double gamma;
	double pitch_prior;
}

struct RigProblem
{
	int t;
	int j;
	int m;
	int v;
	bool known_cameras;
	int* parents;
	int* marker_joint;
	int[] tracked;
	char* fixed;
	int* seen_below;
	int* twist_seen;
	int* twist_held;
	int* chain_child;
	int* chain_branch;
	int[] feet;
	double k0;
	double sigma2;
	double height;
	double[3] root_rest;
	double* w;
	double* offsets;
	double* reach;
	double* twist_axis;
	double* chain_reach;
	double[] lateral;   // [j, anchor, rest, w]
	double[] limits;    // [joint, frame 4, (has, lo, hi) 3]
	FitView[] views;
	// The parameters and their gradients, in fit.js's one buffer each.
	double* theta;
	double* root;
	double* marks;
	double* log_s;
	double* shift;
	double* cam;
	double* g_theta;
	double* g_root;
	double* g_marks;
	double* g_log_s;
	double* g_shift;
	double* g_cam;
	long params;
	// Scratch: forward kinematics' results.
	double[] r;
	double[] gm;
	double[] pos;
	double[] mpos;
	double[] g_pos;
	double[] g_m;
	double[] root_tau;
	double[] s;
	double[] xs;
	double[] a;
	double[] e2;
	double[] ys;
	double limit_scale;
	double observed;
}

faultdef FIT_BAD_LAYOUT;

macro double fit_nan() => bitcast(0x7FF8000000000000UL, double);
macro double fit_inf() => bitcast(0x7FF0000000000000UL, double);

// A typed array's bytes read as doubles or ints, whatever its element type, as
// these verbs always have; anything else is empty.
macro double[] @doubles(value) => (double[])((double*)js::bytes(value).ptr)[:js::bytes(value).len / 8];
macro int[] @ints(value) => (int[])((int*)js::bytes(value).ptr)[:js::bytes(value).len / 4];

<*
 The problem from fit.js's packing: `ints` and `reals` RigFit.native's layout,
 `obs` every view's x, y, w, `params` and `grads` its two buffers.
*>
fn void? RigProblem.read(&self, int[] ints, double[] reals, double[] obs, double[] params, double[] grads)
{
	if (ints.len < 9) return FIT_BAD_LAYOUT~;
	*self = {};
	long t = self.t = ints[0];
	long j = self.j = ints[1];
	long m = self.m = ints[2];
	long v = self.v = ints[3];
	long tracked = ints[4];
	long feet = ints[5];
	long lateral = ints[6];
	long limits = ints[7];
	self.known_cameras = ints[8] != 0;
	if (t <= 0 || j <= 0 || m < 0 || v <= 0) return FIT_BAD_LAYOUT~;
	long at = 9;
	long need = at + (long)j * 7 + (long)m * 2 + tracked + feet;
	if (ints.len < need) return FIT_BAD_LAYOUT~;
	int* base = ints.ptr;
	self.parents = base + at; at += j;
	self.marker_joint = base + at; at += m;
	self.tracked = base[at:tracked]; at += tracked;
	int* fixed = base + at; at += m;
	self.seen_below = base + at; at += j;
	self.twist_seen = base + at; at += j;
	self.twist_held = base + at; at += j;
	self.chain_child = base + at; at += j;
	self.chain_branch = base + at; at += j;
	self.feet = base[at:feet]; at += feet;
	self.fixed = mem::new_array(char, (long)math::max(m, 1L));
	for (long k = 0; k < m; k++) self.fixed[k] = fixed[k] != 0 ? 1 : 0;

	long nw = FIT_WEIGHTS;
	long rneed = 6 + nw + (long)v * 2 + (long)j * 8 + (long)lateral * 4 + (long)limits * 14;
	if (reals.len < rneed) return FIT_BAD_LAYOUT~;
	self.k0 = reals[0];
	self.sigma2 = reals[1];
	self.height = reals[2];
	self.root_rest = { reals[3], reals[4], reals[5] };
	long r = 6;
	self.w = reals.ptr + r; r += nw;
	double* pitch = reals.ptr + r; r += v;
	double* gamma = reals.ptr + r; r += v;
	self.offsets = reals.ptr + r; r += (long)j * 3;
	self.reach = reals.ptr + r; r += j;
	self.twist_axis = reals.ptr + r; r += (long)j * 3;
	self.chain_reach = reals.ptr + r; r += j;
	self.lateral = reals[r:(long)lateral * 4]; r += (long)lateral * 4;
	self.limits = reals[r:(long)limits * 14];

	self.views = mem::new_array(FitView, v);
	long o = 0;
	for (long k = 0; k < v; k++)
	{
		if (ints.len < at + 2) return FIT_BAD_LAYOUT~;
		long p = ints[at];
		long h = ints[at + 1];
		at += 2;
		if (ints.len < at + (long)p * 2) return FIT_BAD_LAYOUT~;
		long n = (long)t * p * h;
		if (obs.len < o + n * 3) return FIT_BAD_LAYOUT~;
		self.views[k] = { .p = (int)p, .h = (int)h, .points = base + at, .x = obs.ptr + o, .y = obs.ptr + o + n, .w = obs.ptr + o + 2 * n,
			.gamma = gamma[k], .pitch_prior = pitch[k] };
		at += (long)p * 2;
		o += n * 3;
	}

	long tj = (long)t * j;
	self.params = tj * 3 + (long)t * 3 + (long)m * 3 + (long)v * t + (long)v * t * 2 + (long)v * 2;
	if (params.len != self.params || grads.len != self.params) return FIT_BAD_LAYOUT~;
	double* p = params.ptr;
	double* g = grads.ptr;
	self.theta = p; self.g_theta = g; p += tj * 3; g += tj * 3;
	self.root = p; self.g_root = g; p += (long)t * 3; g += (long)t * 3;
	self.marks = p; self.g_marks = g; p += (long)m * 3; g += (long)m * 3;
	self.log_s = p; self.g_log_s = g; p += (long)v * t; g += (long)v * t;
	self.shift = p; self.g_shift = g; p += (long)v * t * 2; g += (long)v * t * 2;
	self.cam = p; self.g_cam = g;

	self.r = mem::new_array(double, tj * 9);
	self.gm = mem::new_array(double, tj * 9);
	self.pos = mem::new_array(double, tj * 3);
	self.mpos = mem::new_array(double, math::max(t * m * 3, 1L));
	self.g_pos = mem::new_array(double, tj * 3);
	self.g_m = mem::new_array(double, math::max(t * m * 3, 1L));
	self.root_tau = mem::new_array(double, (long)t * 3);
	self.s = mem::new_array(double, (long)j * 3);
	self.xs = mem::new_array(double, (long)j * 3);
	long most = 1;
	foreach (&view : self.views) most = math::max(most, (long)view.h);
	self.a = mem::new_array(double, most);
	self.e2 = mem::new_array(double, most);
	self.ys = mem::new_array(double, math::max(self.feet.len, (long)1));
	self.limit_scale = 1;
}

fn void RigProblem.free(&self)
{
	if (self.fixed) mem::free(self.fixed);
	if (self.views.ptr) mem::free(self.views);
	double[][12] scratch = { self.r, self.gm, self.pos, self.mpos, self.g_pos, self.g_m, self.root_tau, self.s, self.xs, self.a, self.e2, self.ys };
	foreach (a : scratch) if (a.ptr) mem::free(a);
	*self = {};
}

// Row-major R (9 at `o`) from the rotation vector (x, y, z).
fn void rodrigues(double x, double y, double z, double* r) @local
{
	double t2 = x * x + y * y + z * z;
	double t = math::sqrt(t2);
	double a, b;
	if (t < 1e-6) { a = 1 - t2 / 6; b = 0.5 - t2 / 24; } else { a = math::sin(t) / t; b = (1 - math::cos(t)) / t2; }
	r[0] = 1 - b * (y * y + z * z); r[1] = b * x * y - a * z; r[2] = b * x * z + a * y;
	r[3] = b * x * y + a * z; r[4] = 1 - b * (x * x + z * z); r[5] = b * y * z - a * x;
	r[6] = b * x * z - a * y; r[7] = b * y * z + a * x; r[8] = 1 - b * (x * x + y * y);
}

// J_l(θ)ᵀ g into out: g - a θ × g + b θ × (θ × g).
fn void left_jacobian_t(double x, double y, double z, double g0, double g1, double g2, double* out) @local
{
	double t2 = x * x + y * y + z * z;
	double t = math::sqrt(t2);
	double a, b;
	if (t < 1e-6) { a = 0.5 - t2 / 24; b = 1.0 / 6 - t2 / 120; } else { a = (1 - math::cos(t)) / t2; b = (t - math::sin(t)) / (t2 * t); }
	double c0 = y * g2 - z * g1;
	double c1 = z * g0 - x * g2;
	double c2 = x * g1 - y * g0;
	double d0 = y * c2 - z * c1;
	double d1 = z * c0 - x * c2;
	double d2 = x * c1 - y * c0;
	out[0] = g0 - a * c0 + b * d0;
	out[1] = g1 - a * c1 + b * d1;
	out[2] = g2 - a * c2 + b * d2;
}

macro double hypot3(double x, double y, double z) => math::sqrt(x * x + y * y + z * z);

// How far θ is past a limit (fit.js limitExcess): Σ of the excesses².
fn double limit_excess(double* lim, double x, double y, double z) @local
{
	double t = hypot3(x, y, z);
	double k = t < 1e-12 ? 0.5 : math::sin(t / 2) / t;
	double rx = x * k;
	double ry = y * k;
	double rz = z * k;
	double rw = math::cos(t / 2);
	double fx = lim[1];
	double fy = lim[2];
	double fz = lim[3];
	double fw = lim[4];
	double ax = rw * fx + rx * fw + ry * fz - rz * fy;
	double ay = rw * fy - rx * fz + ry * fw + rz * fx;
	double az = rw * fz + rx * fy - ry * fx + rz * fw;
	double aw = rw * fw - rx * fx - ry * fy - rz * fz;
	double mx = fw * ax - fx * aw - fy * az + fz * ay;
	double my = fw * ay + fx * az - fy * aw - fz * ax;
	double mz = fw * az - fx * ay + fy * ax - fz * aw;
	double mw = fw * aw + fx * ax + fy * ay + fz * az;
	if (mw < 0) { mx = -mx; my = -my; mz = -mz; mw = -mw; }
	double reach = math::sqrt(mw * mw + my * my);
	double ty = reach > 1e-6 ? my / reach : 0;
	double tw = reach > 1e-6 ? mw / reach : 1;
	double sx = mx * tw + mz * ty;
	double swz = mz * tw - mx * ty;
	double sw = mw * tw + my * ty;
	double across = math::sqrt(sx * sx + swz * swz);
	double scale = across < 1e-8 ? 2 / math::max(sw, 1e-8) : (2 * c_atan2(across, sw)) / across;
	double twist = 2 * c_atan2(ty, tw);
	if (twist <= -math::PI) twist += 2 * math::PI;
	double e = 0;
	for (long i = 0; i < 3; i++)
	{
		double* r = lim + 5 + i * 3;
		if (r[0] == 0) continue;
		double angle = i == 0 ? sx * scale : i == 1 ? twist : swz * scale;
		if (angle >= r[1] && angle <= r[2]) continue;
		double along = c_fmod(angle - r[1], 2 * math::PI);
		if (along < 0) along += 2 * math::PI;
		double span = r[2] - r[1];
		if (along <= span) continue;
		double d = math::min(along - span, 2 * math::PI - along);
		e += d * d;
	}
	return e;
}

fn void RigProblem.forward(&self)
{
	long tn = self.t;
	long jn = self.j;
	double* rr = self.r.ptr;
	double* gg = self.gm.ptr;
	double* pos = self.pos.ptr;
	double* theta = self.theta;
	for (long t = 0; t < tn; t++)
	{
		for (long j = 0; j < jn; j++)
		{
			long tj = (long)t * jn + j;
			long o9 = tj * 9;
			long o3 = tj * 3;
			rodrigues(theta[o3], theta[o3 + 1], theta[o3 + 2], rr + o9);
			long p = self.parents[j];
			if (p < 0)
			{
				for (long k = 0; k < 9; k++) gg[o9 + k] = rr[o9 + k];
				pos[o3] = self.root[t * 3]; pos[o3 + 1] = self.root[t * 3 + 1]; pos[o3 + 2] = self.root[t * 3 + 2];
				continue;
			}
			long q9 = ((long)t * jn + p) * 9;
			long q3 = ((long)t * jn + p) * 3;
			for (long r = 0; r < 3; r++)
			{
				double g0 = gg[q9 + r * 3];
				double g1 = gg[q9 + r * 3 + 1];
				double g2 = gg[q9 + r * 3 + 2];
				gg[o9 + r * 3] = g0 * rr[o9] + g1 * rr[o9 + 3] + g2 * rr[o9 + 6];
				gg[o9 + r * 3 + 1] = g0 * rr[o9 + 1] + g1 * rr[o9 + 4] + g2 * rr[o9 + 7];
				gg[o9 + r * 3 + 2] = g0 * rr[o9 + 2] + g1 * rr[o9 + 5] + g2 * rr[o9 + 8];
				pos[o3 + r] = pos[q3 + r] + g0 * self.offsets[j * 3] + g1 * self.offsets[j * 3 + 1] + g2 * self.offsets[j * 3 + 2];
			}
		}
		for (long m = 0; m < self.m; m++)
		{
			long j = self.marker_joint[m];
			long g9 = ((long)t * jn + j) * 9;
			long j3 = ((long)t * jn + j) * 3;
			long m3 = (long)m * 3;
			long x3 = ((long)t * self.m + m) * 3;
			for (long r = 0; r < 3; r++)
			{
				self.mpos[x3 + r] = pos[j3 + r] + gg[g9 + r * 3] * self.marks[m3] + gg[g9 + r * 3 + 1] * self.marks[m3 + 1] + gg[g9 + r * 3 + 2] * self.marks[m3 + 2];
			}
		}
	}
}

<*
 The loss and its gradient into the grads (fit.js evaluate); the terms into
 `terms` (NaN where fit.js has none).
*>
fn double RigProblem.evaluate(&self, double tau, double[FIT_TERMS]* terms)
{
	long tn = self.t;
	long jn = self.j;
	long mn = self.m;
	double k0 = self.k0;
	double sigma2 = self.sigma2;
	double* w = self.w;
	self.forward();
	for (long i = 0; i < self.params; i++) self.g_theta[i] = 0;
	self.g_pos[..] = 0;
	self.g_m[..] = 0;
	self.root_tau[..] = 0;
	for (long i = 0; i < FIT_TERMS; i++) (*terms)[i] = fit_nan();

	// Reprojection, each view through its own camera.
	double reproj = 0;
	long count = 0;
	double[] a = self.a;
	double[] e2 = self.e2;
	for (long v = 0; v < self.v; v++)
	{
		FitView* obs = &self.views[v];
		long pn = obs.p;
		long hn = obs.h;
		double yaw = self.cam[2 * v];
		double pitch = self.cam[2 * v + 1];
		double cy = math::cos(yaw);
		double sy = math::sin(yaw);
		double cp = math::cos(pitch);
		double sp = math::sin(pitch);
		double[3] r0 = { cy, 0, sy };
		double[3] r1 = { sp * sy, cp, -sp * cy };
		double[3] dy0 = { -sy, 0, cy };
		double[3] dy1 = { sp * cy, 0, sp * sy };
		double[3] dp1 = { cp * sy, -sp, -cp * cy };
		double norm = 1.0 / ((double)tn * pn);
		for (long t = 0; t < tn; t++)
		{
			long vt = (long)v * tn + t;
			double s = math::exp(self.log_s[vt]);
			double sx = self.shift[vt * 2];
			double syy = self.shift[vt * 2 + 1];
			for (long pi = 0; pi < pn; pi++)
			{
				long joint = obs.points[pi * 2];
				long marker = obs.points[pi * 2 + 1];
				double* xx;
				double* gx;
				if (marker < 0)
				{
					long o = ((long)t * jn + joint) * 3;
					xx = self.pos.ptr + o; gx = self.g_pos.ptr + o;
				}
				else
				{
					long o = ((long)t * mn + marker) * 3;
					xx = self.mpos.ptr + o; gx = self.g_m.ptr + o;
				}
				double x0 = xx[0];
				double x1 = xx[1];
				double x2 = xx[2];
				double q0 = r0[0] * x0 + r0[2] * x2;
				double q1 = r1[0] * x0 + r1[1] * x1 + r1[2] * x2;
				double u0 = s * q0 + sx;
				double u1 = s * q1 + syy;
				double best = -fit_inf();
				double pull = 0;
				bool seen_any = false;
				long base = ((long)t * pn + pi) * hn;
				for (long h = 0; h < hn; h++)
				{
					double wh = obs.w[base + h];
					if (!(wh > 0)) { a[h] = -fit_inf(); continue; }
					seen_any = true;
					if (wh > pull) pull = wh;
					double d0 = u0 - obs.x[base + h];
					double d1 = u1 - obs.y[base + h];
					e2[h] = d0 * d0 + d1 * d1;
					a[h] = (-sigma2 * c_log1p(e2[h] / sigma2)) / tau + math::ln(wh);
					if (a[h] > best) best = a[h];
				}
				if (!seen_any) continue;
				double c = obs.gamma > 0 ? math::pow(pull, obs.gamma) : 1;
				double z = 0;
				for (long h = 0; h < hn; h++) if (a[h] > -fit_inf()) z += math::exp(a[h] - best);
				reproj += -tau * (best + math::ln(z)) * norm * c;
				count++;
				double du0 = 0;
				double du1 = 0;
				for (long h = 0; h < hn; h++)
				{
					if (a[h] == -fit_inf()) continue;
					double wh = math::exp(a[h] - best) / z;
					double k = (wh * 2) / (1 + e2[h] / sigma2) * norm * w[FitWeight.REPROJ.ordinal] * c;
					du0 += k * (u0 - obs.x[base + h]);
					du1 += k * (u1 - obs.y[base + h]);
				}
				self.g_shift[vt * 2] += du0;
				self.g_shift[vt * 2 + 1] += du1;
				self.g_log_s[vt] += du0 * s * q0 + du1 * s * q1;
				self.g_cam[2 * v] += s * (du0 * (dy0[0] * x0 + dy0[2] * x2) + du1 * (dy1[0] * x0 + dy1[2] * x2));
				self.g_cam[2 * v + 1] += s * du1 * (dp1[0] * x0 + dp1[1] * x1 + dp1[2] * x2);
				gx[0] += s * (du0 * r0[0] + du1 * r1[0]);
				gx[1] += s * du1 * r1[1];
				gx[2] += s * (du0 * r0[2] + du1 * r1[2]);
			}
		}
	}
	(*terms)[T_REPROJ] = reproj * w[FitWeight.REPROJ.ordinal];

	// Accelerations of the tracked joints and the markers, in pixels.
	if (tn > 2)
	{
		double c = (w[FitWeight.ACC.ordinal] * k0 * k0) / ((double)(tn - 2) * (double)(jn + mn));
		double s2 = (ACC_SCALE / k0) * (ACC_SCALE / k0);
		double acc = 0;
		for (long pass = 0; pass < 2; pass++)
		{
			double* xx = pass == 0 ? self.pos.ptr : self.mpos.ptr;
			double* gx = pass == 0 ? self.g_pos.ptr : self.g_m.ptr;
			long n = pass == 0 ? (long)jn : (long)mn;
			long which = pass == 0 ? self.tracked.len : (long)mn;
			for (long t = 1; t + 1 < (long)tn; t++)
			{
				for (long k = 0; k < which; k++)
				{
					long j = pass == 0 ? (long)self.tracked[k] : k;
					double[3] aa;
					double a2 = 0;
					for (long d = 0; d < 3; d++)
					{
						long i = j * 3 + d;
						aa[d] = xx[(t + 1) * n * 3 + i] - 2 * xx[t * n * 3 + i] + xx[(t - 1) * n * 3 + i];
						a2 += aa[d] * aa[d];
					}
					// Pseudo-Huber: the square near zero, linear far out.
					double root = math::sqrt(1 + a2 / s2);
					double kk = (2 * c) / root;
					acc += 2 * c * s2 * (root - 1);
					for (long d = 0; d < 3; d++)
					{
						long i = j * 3 + d;
						gx[(t + 1) * n * 3 + i] += kk * aa[d];
						gx[t * n * 3 + i] -= 2 * kk * aa[d];
						gx[(t - 1) * n * 3 + i] += kk * aa[d];
					}
				}
			}
		}
		(*terms)[T_ACC] = acc;
	}

	// The floor: each frame's lowest foot point at 0 (softly lowest), none below.
	if (self.feet.len)
	{
		long fn_ = self.feet.len;
		double tf = 3;
		double floor = 0;
		double wf_ = w[FitWeight.FLOOR.ordinal];
		double[] ys = self.ys;
		for (long t = 0; t < tn; t++)
		{
			double m = fit_inf();
			for (long f = 0; f < fn_; f++) { ys[f] = self.pos[((long)t * jn + self.feet[f]) * 3 + 1] * k0; if (ys[f] < m) m = ys[f]; }
			double z = 0;
			for (long f = 0; f < fn_; f++) z += math::exp(-(ys[f] - m) / tf);
			double low = m - tf * math::ln(z);
			floor += (wf_ * low * low) / tn;
			for (long f = 0; f < fn_; f++)
			{
				double wf = math::exp(-(ys[f] - m) / tf) / z;
				double d = (wf_ * 2 * low * wf) / tn;
				if (ys[f] < 0)
				{
					floor += (wf_ * ys[f] * ys[f]) / ((double)tn * fn_);
					d += (wf_ * 2 * ys[f]) / ((double)tn * fn_);
				}
				self.g_pos[((long)t * jn + self.feet[f]) * 3 + 1] += d * k0;
			}
		}
		(*terms)[T_FLOOR] = floor;
	}

	// In place: the root's spread about its mean (and that mean at rest).
	{
		double[3] mean;
		for (long t = 0; t < tn; t++) for (long d = 0; d < 3; d++) mean[d] += self.root[t * 3 + d] / tn;
		double ip = 0;
		double wi = w[FitWeight.IN_PLACE.ordinal];
		double[3] wd = { 1, 0.1, 1 };
		for (long t = 0; t < tn; t++)
		{
			for (long d = 0; d < 3; d++)
			{
				double vv = (self.root[t * 3 + d] - mean[d]) * k0;
				ip += (wi * wd[d] * vv * vv) / tn;
				self.g_root[t * 3 + d] += ((wi * wd[d] * 2 * vv) / tn) * k0;
			}
		}
		for (long d = 0; d < 3; d += 2)
		{
			double vv = (mean[d] - self.root_rest[d]) * k0;
			ip += wi * vv * vv;
			for (long t = 0; t < tn; t++) self.g_root[t * 3 + d] += ((wi * 2 * vv) / tn) * k0;
		}
		(*terms)[T_IN_PLACE] = ip;
	}

	// Markers stay near their joints.
	{
		double mk = 0;
		double wm = w[FitWeight.MARKS.ordinal];
		double per = (double)math::max(1L, mn);
		for (long i = 0; i < mn * 3; i++)
		{
			if (self.fixed[i / 3]) continue;
			double vv = self.marks[i] * k0;
			mk += (wm * vv * vv) / per;
			self.g_marks[i] += ((wm * 2 * vv) / per) * k0;
		}
		(*terms)[T_MARKS] = mk;
	}

	// Sideways offsets in the root's frame.
	long nl = self.lateral.len / 4;
	if (nl)
	{
		double lat = 0;
		double scale = (w[FitWeight.LATERAL.ordinal] * k0 * k0) / ((double)tn * nl);
		for (long t = 0; t < tn; t++)
		{
			long g9 = (long)t * jn * 9;
			double a0 = self.gm[g9];
			double a1 = self.gm[g9 + 3];
			double a2 = self.gm[g9 + 6];
			for (long l = 0; l < nl; l++)
			{
				double* row = self.lateral.ptr + l * 4;
				long j = (int)row[0];
				long anchor = (int)row[1];
				double rest = row[2];
				double lw = row[3];
				long o = ((long)t * jn + j) * 3;
				long q = ((long)t * jn + anchor) * 3;
				double d0 = self.pos[o] - self.pos[q];
				double d1 = self.pos[o + 1] - self.pos[q + 1];
				double d2 = self.pos[o + 2] - self.pos[q + 2];
				double e = d0 * a0 + d1 * a1 + d2 * a2 - rest;
				lat += scale * lw * e * e;
				double c = 2 * scale * lw * e;
				self.g_pos[o] += c * a0; self.g_pos[o + 1] += c * a1; self.g_pos[o + 2] += c * a2;
				self.g_pos[q] -= c * a0; self.g_pos[q + 1] -= c * a1; self.g_pos[q + 2] -= c * a2;
				self.root_tau[t * 3] += c * (a1 * d2 - a2 * d1);
				self.root_tau[t * 3 + 1] += c * (a2 * d0 - a0 * d2);
				self.root_tau[t * 3 + 2] += c * (a0 * d1 - a1 * d0);
			}
		}
		(*terms)[T_LATERAL] = lat;
	}

	self.backward();
	for (long m = 0; m < mn; m++) if (self.fixed[m]) for (long d = 0; d < 3; d++) self.g_marks[m * 3 + d] = 0;
	self.turn_priors(terms);
	self.camera_priors(terms);
	double total = 0;
	for (long i = 0; i < FIT_TERMS; i++) if ((*terms)[i] == (*terms)[i]) total += (*terms)[i];
	self.observed = count;
	return total;
}

// The joints' and markers' position gradients up the tree.
fn void RigProblem.backward(&self)
{
	long tn = self.t;
	long jn = self.j;
	long mn = self.m;
	double* sv = self.s.ptr;
	double* xv = self.xs.ptr;
	double* pos = self.pos.ptr;
	double* gpos = self.g_pos.ptr;
	double* gg = self.gm.ptr;
	double[3] out;
	for (long t = 0; t < tn; t++)
	{
		for (long j = 0; j < jn; j++)
		{
			long o = ((long)t * jn + j) * 3;
			sv[j * 3] = gpos[o]; sv[j * 3 + 1] = gpos[o + 1]; sv[j * 3 + 2] = gpos[o + 2];
			xv[j * 3] = pos[o + 1] * gpos[o + 2] - pos[o + 2] * gpos[o + 1];
			xv[j * 3 + 1] = pos[o + 2] * gpos[o] - pos[o] * gpos[o + 2];
			xv[j * 3 + 2] = pos[o] * gpos[o + 1] - pos[o + 1] * gpos[o];
		}
		for (long mk = 0; mk < mn; mk++)
		{
			long m = ((long)t * mn + mk) * 3;
			long j = self.marker_joint[mk];
			long g9 = ((long)t * jn + j) * 9;
			double gx = self.g_m[m];
			double gy = self.g_m[m + 1];
			double gz = self.g_m[m + 2];
			double x = self.mpos[m];
			double y = self.mpos[m + 1];
			double z = self.mpos[m + 2];
			sv[j * 3] += gx; sv[j * 3 + 1] += gy; sv[j * 3 + 2] += gz;
			xv[j * 3] += y * gz - z * gy; xv[j * 3 + 1] += z * gx - x * gz; xv[j * 3 + 2] += x * gy - y * gx;
			for (long d = 0; d < 3; d++) self.g_marks[mk * 3 + d] += gg[g9 + d] * gx + gg[g9 + 3 + d] * gy + gg[g9 + 6 + d] * gz;
		}
		for (long j = jn - 1; j > 0; j--)
		{
			long p = self.parents[j];
			for (long d = 0; d < 3; d++) { sv[p * 3 + d] += sv[j * 3 + d]; xv[p * 3 + d] += xv[j * 3 + d]; }
		}
		for (long j = 0; j < jn; j++)
		{
			long o = ((long)t * jn + j) * 3;
			double s0 = sv[j * 3];
			double s1 = sv[j * 3 + 1];
			double s2 = sv[j * 3 + 2];
			double p0 = pos[o];
			double p1 = pos[o + 1];
			double p2 = pos[o + 2];
			double t0 = xv[j * 3] - (p1 * s2 - p2 * s1);
			double t1 = xv[j * 3 + 1] - (p2 * s0 - p0 * s2);
			double t2 = xv[j * 3 + 2] - (p0 * s1 - p1 * s0);
			if (j == 0) { t0 += self.root_tau[t * 3]; t1 += self.root_tau[t * 3 + 1]; t2 += self.root_tau[t * 3 + 2]; }
			long par = self.parents[j];
			double l0 = t0;
			double l1 = t1;
			double l2 = t2;
			if (par >= 0)
			{
				long q9 = ((long)t * jn + par) * 9;
				l0 = gg[q9] * t0 + gg[q9 + 3] * t1 + gg[q9 + 6] * t2;
				l1 = gg[q9 + 1] * t0 + gg[q9 + 4] * t1 + gg[q9 + 7] * t2;
				l2 = gg[q9 + 2] * t0 + gg[q9 + 5] * t1 + gg[q9 + 8] * t2;
			}
			else
			{
				self.g_root[t * 3] += s0; self.g_root[t * 3 + 1] += s1; self.g_root[t * 3 + 2] += s2;
			}
			left_jacobian_t(self.theta[o], self.theta[o + 1], self.theta[o + 2], l0, l1, l2, &out);
			self.g_theta[o] += out[0]; self.g_theta[o + 1] += out[1]; self.g_theta[o + 2] += out[2];
		}
	}
}

// Twist, rest, chain, the turns' smoothness, heading and limits.
fn void RigProblem.turn_priors(&self, double[FIT_TERMS]* terms)
{
	long tn = self.t;
	long jn = self.j;
	double k0 = self.k0;
	double* w = self.w;
	double* theta = self.theta;
	double* g = self.g_theta;
	double twist = 0;
	double rest = 0;
	double chain = 0;
	double tacc = 0;
	double norm = 1.0 / ((double)tn * jn);
	for (long j = 0; j < jn; j++)
	{
		double l = math::max(self.reach[j], 0.05 * self.height) * k0;
		double l2 = l * l;
		long seen = self.seen_below[j];
		double wr = (seen == 0 ? w[FitWeight.UNSEEN.ordinal] : seen == 1 ? w[FitWeight.ONE_MARKER.ordinal] : w[FitWeight.REST.ordinal]) * l2 * norm;
		double wt = (self.twist_seen[j] ? w[FitWeight.TWIST_MARKED.ordinal] : w[FitWeight.TWIST.ordinal]) * l2 * norm;
		double* b = self.twist_axis + j * 3;
		long c = self.chain_child[j];
		for (long t = 0; t < tn; t++)
		{
			long o = ((long)t * jn + j) * 3;
			for (long d = 0; d < 3; d++)
			{
				rest += wr * theta[o + d] * theta[o + d];
				g[o + d] += 2 * wr * theta[o + d];
			}
			if (self.twist_held[j] && wt > 0)
			{
				double tw = theta[o] * b[0] + theta[o + 1] * b[1] + theta[o + 2] * b[2];
				twist += wt * tw * tw;
				for (long d = 0; d < 3; d++) g[o + d] += 2 * wt * tw * b[d];
			}
			if (c >= 0)
			{
				double lc = math::max(self.chain_reach[j], 0.05 * self.height) * k0;
				double wc = (self.chain_branch[j] ? w[FitWeight.CHAIN_BRANCH.ordinal] : w[FitWeight.CHAIN.ordinal]) * lc * lc * norm;
				long q = ((long)t * jn + c) * 3;
				for (long d = 0; d < 3; d++)
				{
					double vv = theta[q + d] - theta[o + d];
					chain += wc * vv * vv;
					g[q + d] += 2 * wc * vv;
					g[o + d] -= 2 * wc * vv;
				}
			}
		}
		if (tn > 2)
		{
			double wa = (w[FitWeight.THETA_ACC.ordinal] * l2) / ((double)(tn - 2) * (double)jn);
			double s2 = (ACC_SCALE * ACC_SCALE) / l2;
			long n = (long)jn * 3;
			for (long t = 1; t < tn - 1; t++)
			{
				double[3] aa;
				double a2 = 0;
				for (long d = 0; d < 3; d++)
				{
					long i = ((long)t * jn + j) * 3 + d;
					aa[d] = theta[i + n] - 2 * theta[i] + theta[i - n];
					a2 += aa[d] * aa[d];
				}
				double root = math::sqrt(1 + a2 / s2);
				double kk = (2 * wa) / root;
				tacc += 2 * wa * s2 * (root - 1);
				for (long d = 0; d < 3; d++)
				{
					long i = ((long)t * jn + j) * 3 + d;
					g[i + n] += kk * aa[d];
					g[i] -= 2 * kk * aa[d];
					g[i - n] += kk * aa[d];
				}
			}
		}
	}
	// Heading: the root's mean turn about the vertical held at none.
	{
		double l = math::max(self.reach[0], 0.05 * self.height) * k0;
		double mean = 0;
		for (long t = 0; t < tn; t++) mean += theta[(long)t * jn * 3 + 1] / tn;
		double hw = w[FitWeight.HEADING.ordinal] * l * l;
		(*terms)[T_HEADING] = hw * mean * mean;
		for (long t = 0; t < tn; t++) g[(long)t * jn * 3 + 1] += (2 * hw * mean) / tn;
	}
	// Limits, by central differences.
	double limit = 0;
	long nl = self.limits.len / 14;
	for (long li = 0; li < nl; li++)
	{
		double* lim = self.limits.ptr + li * 14;
		long j = (int)lim[0];
		double l = math::max(self.reach[j], 0.05 * self.height) * k0;
		double lw = w[FitWeight.LIMIT.ordinal] * self.limit_scale * l * l * norm;
		double h = 1e-6;
		for (long t = 0; t < tn; t++)
		{
			long o = ((long)t * jn + j) * 3;
			double x = theta[o];
			double y = theta[o + 1];
			double z = theta[o + 2];
			double e = limit_excess(lim, x, y, z);
			if (e == 0) continue;
			limit += lw * e;
			g[o] += (lw * (limit_excess(lim, x + h, y, z) - limit_excess(lim, x - h, y, z))) / (2 * h);
			g[o + 1] += (lw * (limit_excess(lim, x, y + h, z) - limit_excess(lim, x, y - h, z))) / (2 * h);
			g[o + 2] += (lw * (limit_excess(lim, x, y, z + h) - limit_excess(lim, x, y, z - h))) / (2 * h);
		}
	}
	(*terms)[T_TWIST] = twist;
	(*terms)[T_REST] = rest;
	(*terms)[T_CHAIN] = chain;
	(*terms)[T_THETA_ACC] = tacc;
	(*terms)[T_LIMIT] = limit;
}

// Each view's pitch near its prior, its scale and aim smooth.
fn void RigProblem.camera_priors(&self, double[FIT_TERMS]* terms)
{
	long tn = self.t;
	double* w = self.w;
	double level = 0;
	double follow = 0;
	for (long v = 0; v < self.v; v++)
	{
		double dp = self.cam[2 * v + 1] - self.views[v].pitch_prior;
		level += w[FitWeight.LEVEL.ordinal] * dp * dp;
		self.g_cam[2 * v + 1] += 2 * w[FitWeight.LEVEL.ordinal] * dp;
		if (tn <= 2) continue;
		double cs = (w[FitWeight.FOLLOW.ordinal] * 1e6) / (double)(tn - 2);
		double ch = w[FitWeight.FOLLOW.ordinal] / (double)(tn - 2);
		long o = v * tn;
		for (long t = o + 1; t < o + tn - 1; t++)
		{
			double aa = self.log_s[t + 1] - 2 * self.log_s[t] + self.log_s[t - 1];
			follow += cs * aa * aa;
			self.g_log_s[t + 1] += 2 * cs * aa; self.g_log_s[t] -= 4 * cs * aa; self.g_log_s[t - 1] += 2 * cs * aa;
			for (long d = 0; d < 2; d++)
			{
				double bb = self.shift[(t + 1) * 2 + d] - 2 * self.shift[t * 2 + d] + self.shift[(t - 1) * 2 + d];
				follow += ch * bb * bb;
				self.g_shift[(t + 1) * 2 + d] += 2 * ch * bb; self.g_shift[t * 2 + d] -= 4 * ch * bb; self.g_shift[(t - 1) * 2 + d] += 2 * ch * bb;
			}
		}
	}
	(*terms)[T_LEVEL] = level;
	(*terms)[T_FOLLOW] = follow;
}

<*
 One Adam step (fit.js Adam.step) over the six parameter groups, each with
 its learning rate in `lr` (0 holds the group).
*>
fn void RigProblem.adam_step(&self, double* am, double* av, double* lr, double t, double scale)
{
	double c1 = 1 - math::pow(0.9, t);
	double c2 = 1 - math::pow(0.999, t);
	long[6] sizes = { (long)self.t * self.j * 3, (long)self.t * 3, (long)self.m * 3, (long)self.v * self.t, (long)self.v * self.t * 2, (long)self.v * 2 };
	long at = 0;
	for (long k = 0; k < 6; k++)
	{
		long n = sizes[k];
		defer at += n;
		if (!lr[k]) continue;
		double rate = lr[k] * scale;
		for (long i = at; i < at + n; i++)
		{
			double d = self.g_theta[i];
			am[i] = 0.9 * am[i] + 0.1 * d;
			av[i] = 0.999 * av[i] + 0.001 * d * d;
			self.theta[i] -= (rate * (am[i] / c1)) / (math::sqrt(av[i] / c2) + 1e-8);
		}
	}
}

fn js::Value? fit_answer(js::Engine* js, double[FIT_TERMS]* terms, double loss, double observed, double adam_t, double limit_scale) @local
{
	double[FIT_ANSWER] out;
	for (long i = 0; i < FIT_TERMS; i++) out[i] = (*terms)[i];
	out[FIT_TERMS] = loss;
	out[FIT_TERMS + 1] = observed;
	out[FIT_TERMS + 2] = adam_t;
	out[FIT_TERMS + 3] = limit_scale;
	return js.typed_copy(FLOAT64, ((char*)&out)[:FIT_ANSWER * 8]);
}

<*
 `rigFitEvaluate(ints, reals, obs, params, grads, tau, limitScale)`: the loss's
 gradient into grads; answers the terms, the loss and the count.
*>
fn js::Value? js_rig_fit_evaluate(js::Engine* js, js::Value ints, js::Value reals, js::Value obs, js::Value params, js::Value grads,
	double tau, double limit_scale)
{
	RigProblem p;
	defer p.free();
	if (catch p.read(@ints(ints), @doubles(reals), @doubles(obs), @doubles(params), @doubles(grads)))
	{
		return js.fail("rigFitEvaluate: the arrays do not match RigFit.native's layout")~;
	}
	p.limit_scale = limit_scale;
	double[FIT_TERMS] terms;
	double loss = p.evaluate(tau, &terms);
	return fit_answer(js, &terms, loss, p.observed, 0, p.limit_scale);
}

<*
 `rigFitSteps(ints, reals, obs, params, grads, adamM, adamV, lr, from, to,
 iterations, tau0, tau1, adamT, wide?)`: fit.js RigFit.run's iterations from..to.
*>
fn js::Value? js_rig_fit_steps(js::Engine* js, js::Value ints, js::Value reals, js::Value obs, js::Value params, js::Value grads,
	js::Value adam_m, js::Value adam_v, js::Value rates, long from, long to, long iterations, double tau0, double tau1, double adam_t,
	Opt {double} wide_arg)
{
	RigProblem p;
	defer p.free();
	if (catch p.read(@ints(ints), @doubles(reals), @doubles(obs), @doubles(params), @doubles(grads)))
	{
		return js.fail("rigFitSteps: the arrays do not match RigFit.native's layout")~;
	}
	double[] am = @doubles(adam_m);
	double[] av = @doubles(adam_v);
	double[] lr = @doubles(rates);
	if (am.len != p.params || av.len != p.params || lr.len != 6)
	{
		return js.fail("rigFitSteps: Adam's moments are the params' size and lr has 6 rates")~;
	}
	double wide = wide_arg.or(1.0);
	double sigma2 = p.sigma2;
	double[FIT_TERMS] terms;
	for (long i = 0; i < FIT_TERMS; i++) terms[i] = fit_nan();
	double loss = 0;
	for (long it = from; it < to; it++)
	{
		double tau = tau0 * math::pow(tau1 / tau0, math::min(1.0, it / (0.6 * iterations)));
		p.limit_scale = math::min(1.0, it / (0.5 * iterations));
		p.sigma2 = sigma2 * math::pow(wide, 2 * (1 - math::min(1.0, it / (0.5 * iterations))));
		loss = p.evaluate(tau, &terms);
		if (it == iterations - 1) break;
		adam_t += 1;
		p.adam_step(am.ptr, av.ptr, lr.ptr, adam_t, 0.5 * (1 + math::cos((math::PI * it) / iterations)));
	}
	return fit_answer(js, &terms, loss, p.observed, adam_t, p.limit_scale);
}
