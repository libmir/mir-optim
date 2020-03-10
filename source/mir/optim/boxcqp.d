/++
$(H1 Bound Constrained Convex Quadratic Problem Solver)

Paper: $(HTTP www.cse.uoi.gr/tech_reports/publications/boxcqp.pdf,
    BOXCQP: AN ALGORITHM FOR BOUND CONSTRAINED CONVEX QUADRATIC PROBLEMS)

Copyright: Copyright © 2020, Symmetry Investments & Kaleidic Associates Advisory Limited
Authors:   Ilya Yaroshenko

Macros:
NDSLICE = $(REF_ALTTEXT $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
+/
module mir.optim.boxcqp;

import mir.ndslice.slice: Slice, Canonical;
import lapack: lapackint;
import mir.math.common: fmin, fmax, sqrt, fabs;

/++
BOXCQP Exit Status
+/
enum BoxQPStatus
{
    ///
    solved,
    ///
    numericError,
    ///
    maxIterations,
}

/++
+/
@safe pure nothrow @nogc
size_t boxQPWorkLength(size_t n)
{
    return n ^^ 2 * 2 + n * 8;
}

/++
BOXCQP Algorithm Settings
+/
struct BoxQPSettings(T)
    if (is(T == float) || is(T == double))
{
    /++
    Relative active constraints tolerance.
    +/
    T relTolerance = T.epsilon.sqrt;
    /++
    Absolute active constraints tolerance.
    +/
    T absTolerance = T.epsilon.sqrt;
    /++
    Maximal iterations allowed. `0` is used for default value equals to `10 * N + 100`.
    +/
    uint maxIterations = 0;
}

/++
Solves:
    `argmin_x(xPx + qx) : l <= x <= u`
Params:
    P = Positive-definite Matrix, NxN
    q = Linear component, N
    l = Lower bounds in [-inf, +inf), N
    u = Upper bounds in (-inf, +inf], N
    x = solutoin, N
    settings = Iteration settings (optional)
+/
@safe pure nothrow @nogc
BoxQPStatus solveBoxQP(T)(
    Slice!(const(T)*, 2, Canonical) P,
    Slice!(const(T)*) q,
    Slice!(const(T)*) l,
    Slice!(const(T)*) u,
    Slice!(T*) x,
    BoxQPSettings!T settings = BoxQPSettings!T.init,
    )
    if (is(T == float) || is(T == double))
{
    import mir.ndslice.allocation: rcslice;
    auto n = q.length;
    auto work = rcslice!T(boxQPWorkLength(n));
    auto bwork = rcslice!byte(n);
    auto iwork = rcslice!lapackint(n);
    return solveBoxQP(settings, P, q, l, u, x, work.lightScope, iwork.lightScope, bwork.lightScope);
}

/++
Solves:
    `argmin_x(xPx + qx) : l <= x <= u`
Params:
    settings = Iteration settings
    P = Positive-definite Matrix, NxN
    q = Linear component, N
    l = Lower bounds in [-inf, +inf), N
    u = Upper bounds in (-inf, +inf], N
    x = solutoin, N
    work = workspace, boxQPWorkLength(N)
    iwork = integer workspace, N
    bwork = byte workspace, N
+/
@safe pure nothrow @nogc
BoxQPStatus solveBoxQP(T)(
    ref const BoxQPSettings!T settings,
    Slice!(const(T)*, 2, Canonical) P,
    Slice!(const(T)*) q,
    Slice!(const(T)*) l,
    Slice!(const(T)*) u,
    Slice!(T*) x,
    Slice!(T*) work,
    Slice!(lapackint*) iwork,
    Slice!(byte*) bwork,
)
    if (is(T == float) || is(T == double))
in {
    auto n = q.length;
    assert(P.length!0 == n);
    assert(P.length!1 == n);
    assert(q.length == n);
    assert(l.length == n);
    assert(u.length == n);
    assert(x.length == n);
    assert(iwork.length == n);
    assert(bwork.length == n);
    assert(work.length >= boxQPWorkLength(n));
}
do {
    import mir.blas: dot;
    import mir.lapack: posvx;
    import mir.ndslice.mutation: copyMinor;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: canonical;

    enum Flag : byte
    {
        l = -1,
        s = 0,
        u = 1,
    }

    auto n = q.length;
    if (n == 0)
        return BoxQPStatus.solved;

    auto flags = (()@trusted=>(cast(Flag*)bwork.ptr).sliced(n))();

    auto maxIterations = cast()settings.maxIterations;
    if (!maxIterations)
        maxIterations = cast(uint)n * 10 + 100; // fix

    auto la  = work[0 .. n]; work = work[n .. $];
    auto mu  = work[0 .. n]; work = work[n .. $];

    {
        auto buffer = work;
        auto scaling = buffer[0 .. n]; buffer = buffer[n .. $];
        auto b = buffer[0 .. n]; buffer = buffer[n .. $];
        auto lapackWorkSpace = buffer[0 .. n * 3]; buffer = buffer[n * 3 .. $];
        auto A = buffer[0 .. n ^^ 2].sliced(n, n); buffer = buffer[n ^^ 2 .. $];
        auto F = buffer[0 .. n ^^ 2].sliced(n, n); buffer = buffer[n ^^ 2 .. $];

        A[] = P;
        b[] = -q;

        char equed;
        T rcond, ferr, berr;
        auto info = posvx('E', 'U',
            A.canonical,
            F.canonical,
            equed,
            scaling,
            b,
            x,
            rcond,
            ferr,
            berr,
            lapackWorkSpace,
            iwork.lightScope);

        if (info != 0 && info != n + 1)
            return BoxQPStatus.numericError;
    }

    la[] = 0;
    mu[] = 0;

    foreach (i; 0 .. n)
        if (!(l[i] <= x[i] && x[i] <= u[i]))
            goto MainLoop;
    return BoxQPStatus.solved;

    MainLoop: foreach (step; 0 .. maxIterations)
    {
        {
            size_t s;

            with(settings) foreach (i; 0 .. n)
            {
                auto xl = x[i] - l[i];
                auto ux = u[i] - x[i];
                if (xl < 0 || xl < relTolerance.fmax(absTolerance * l[i].fabs) && la[i] >= 0)
                {
                    flags[i] = Flag.l;
                    x[i] = l[i];
                    mu[i] = 0;
                }
                else
                if (ux < 0 || ux < relTolerance.fmax(absTolerance * u[i].fabs) && mu[i] >= 0)
                {
                    flags[i] = Flag.u;
                    x[i] = u[i];
                    la[i] = 0;
                }
                else
                {
                    flags[i] = Flag.s;
                    iwork[s++] = cast(lapackint)i;
                    mu[i]  = 0;
                    la[i]  = 0;
                }
            }

            {
                auto SIWorkspace = iwork.lightScope[0 .. s];
                auto buffer = work;
                auto scaling = buffer[0 .. s]; buffer = buffer[s .. $];
                auto sX = buffer[0 .. s]; buffer = buffer[s .. $];
                auto b = buffer[0 .. s]; buffer = buffer[s .. $];
                auto lapackWorkSpace = buffer[0 .. s * 3]; buffer = buffer[s * 3 .. $];
                auto A = buffer[0 .. s ^^ 2].sliced(s, s); buffer = buffer[s ^^ 2 .. $];
                auto F = buffer[0 .. s ^^ 2].sliced(s, s); buffer = buffer[s ^^ 2 .. $];

                copyMinor(P, A, SIWorkspace, SIWorkspace);

                foreach (ii, i; SIWorkspace.field)
                {
                    auto Pi = P[i];
                    T sum = q[i];
                    foreach (j; 0 .. n)
                    {
                        if (flags[j] == Flag.l)
                            sum += Pi[j] * l[j];
                        else
                        if (flags[j] == Flag.u)
                            sum += Pi[j] * u[j];
                    }
                    b[ii] = -sum;
                }

                {
                    char equed;
                    T rcond, ferr, berr;
                    auto info = posvx('E', 'U',
                        A.canonical,
                        F.canonical,
                        equed,
                        scaling,
                        b,
                        sX,
                        rcond,
                        ferr,
                        berr,
                        lapackWorkSpace,
                        SIWorkspace);
                    
                    if (info != 0 && info != s + 1)
                        return BoxQPStatus.numericError;
                }

                size_t ii;
                foreach (i; 0 .. n) if (flags[i] == Flag.s)
                    x[i] = sX[ii++];
            }
        }

        foreach (i; 0 .. n) if (flags[i] == Flag.l)
            la[i] = dot!T(P[i], x) + q[i];

        foreach (i; 0 .. n) if (flags[i] == Flag.u)
            mu[i] = -(dot!T(P[i], x) + q[i]);

        foreach (i; 0 .. n)
        {
            final switch (flags[i])
            {
                case Flag.l: if (la[i] >= 0) continue; continue MainLoop;
                case Flag.u: if (mu[i] >= 0) continue; continue MainLoop;
                case Flag.s: if (x[i] >= l[i] && x[i] <= u[i]) continue; continue MainLoop;
            }
        }

        foreach (i; 0 .. n)
            x[i] = x[i].fmin(u[i]).fmax(l[i]);
        return BoxQPStatus.solved;
    }

    return BoxQPStatus.maxIterations;
}

///
version(mir_optim_test)
unittest
{
    import mir.ndslice;

    auto P = [
        [ 2.0, -1, 0],
        [-1.0, 2, -1],
        [ 0.0, -1, 2],
    ].fuse.canonical;

    auto q = [3.0, -7, 5].sliced;
    auto l = [-100.0, -2, 1].sliced;
    auto u = [100.0, 2, 1].sliced;
    auto x = slice!double(q.length);

    solveQP(P, q, l, u, x);
    assert(x == [-0.5, 2, 1]);
}
