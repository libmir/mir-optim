/++
+/
module mir.optim.qp;

import mir.ndslice;
import lapack: lapackint;
import mir.optim.internal;

import mir.math.sum;
import mir.utility: min, max;
import mir.math.common: fmin, fmax, fabs, sqrt;
import mir.math.constant: GoldenRatio;
import mir.blas: gemv;
import mir.internal.memory;
import mir.blas;
import mir.lapack;
import cblas;

/++
+/
struct QPSettings(T)
    if (is(T == double) || is(T == float))
{
    /++
    +/
    T epsAbs = T.epsilon.sqrt.sqrt;
    /++
    +/
    T epsRel = T.epsilon.sqrt.sqrt;
    /++
    +/
    T epsInfeasibility = T.epsilon.sqrt.sqrt;
    /++
    +/
    T initialRho = 1;
    /++
    +/
    T minRho = T.epsilon.sqrt;
    /++
    +/
    T maxRho = 1 / T.epsilon.sqrt; 
    /++
    +/
    T sigma = T.epsilon.sqrt;
    /++
    +/
    T alpha = GoldenRatio;
    /++
    +/
    int minAdaptiveRhoAge = 10;
    /++
    +/
    int maxRhoUpdates = 40;
    /++
    +/
    int maxIterations = 4000;
}

/++
+/
enum QPTerminationCriteria
{
    ///
    numericError = -1,
    ///
    none,
    ///
    solved,
    ///
    primalInfeasibility,
    ///
    dualInfeasibility,
    ///
    maxIterations,
}

/++
+/
struct QPResult(T)
    if (is(T == double) || is(T == float))
{
    ///
    QPTerminationCriteria criteria;
    ///
    int iterations;
    ///
    int rhoUpdates;
    ///
    T objectiveValue;
    ///
    T rho = 0;
}

/++
+/
struct QPWorkspace(T)
    if (is(T == double) || is(T == float))
{
    private alias Type = T;
    /++
    `m x n`
    +/
    Slice!(T*, 2) A;
    /++
    `n x n`
    +/
    Slice!(T*, 2) P;
    /++
    `n`
    +/
    Slice!(T*) q;
    /++
    `n`
    +/
    Slice!(T*) l;
    /++
    `n`
    +/
    Slice!(T*) u;
    /++
    `n`
    +/
    Slice!(T*) x;
    /++
    `m`
    +/
    Slice!(T*) y;
    /++
    `m`
    +/
    Slice!(T*) z;
    /++
    `>= qpWorkLength`
    +/
    Slice!(T*) work;
    /++
    `n`
    +/
    Slice!(lapackint*) iwork;

    ///
    void reset()
    {
        x[] = 0;
        y[] = 0;
    }

    ///
    this(size_t m, size_t n, bool useP, bool useA, bool allowPolish = true)
    {
        if (useA)
        {
            A = [m, n].slice!T(0);
            A.diagonal[] = 1;
        }
        else
        {
            assert(n == m);
        }
        if (useP)
        {
            P = [n, n].slice!T(0);
            P.diagonal[] = 1;
        }
        q = [n].slice!T(0);
        l = [n].slice!T(-T.infinity);
        u = [n].slice!T(+T.infinity);
        x = [n].slice!T(0);
        y = [m].slice!T(0);
        z = [m].slice!T(0);
        iwork = [n].slice!lapackint;
        work = [qpWorkLength(m, n, useP, useA, allowPolish)].slice!T;
    }
}

/++
+/
struct RCQPWorkspace(T)
    if (is(T == double) || is(T == float))
{
    import mir.rc.array;

    private alias Type = T;

    /++
    `m x n`
    +/
    RCArray!T A;
    /++
    `n x n`
    +/
    RCArray!T P;
    /++
    `n`
    +/
    RCArray!T q;
    /++
    `n`
    +/
    RCArray!T l;
    /++
    `n`
    +/
    RCArray!T u;
    /++
    `n`
    +/
    RCArray!T x;
    /++
    `m`
    +/
    RCArray!T y;
    /++
    `m`
    +/
    RCArray!T z;
    /++
    `>= qpWorkLength`
    +/
    RCArray!T work;
    /++
    `n`
    +/
    RCArray!lapackint iwork;

    ///
    void reset()
    {
        x[][] = 0;
        y[][] = 0;
    }

    ///
    this(size_t m, size_t n, bool useP, bool useA, bool allowPolish = true)
    {
        if (useA)
        {
            A = RCArray!T(m * n);
            A[][] = 0;
            A[].sliced(m, n).diagonal[] = 1;
        }
        else
        {
            assert(n == m);
        }
        if (useP)
        {
            P = RCArray!T(n * n);
            P[][] = 0;
            P[].sliced(n, n).diagonal[] = 1;
        }
        q = RCArray!T(n); q[][] = 0;
        l = RCArray!T(n); l[][] = -T.infinity;
        u = RCArray!T(n); u[][] = +T.infinity;
        x = RCArray!T(n); x[][] = 0;
        y = RCArray!T(m); y[][] = 0;
        z = RCArray!T(m); z[][] = 0;
        iwork = RCArray!lapackint(n);
        work = RCArray!T(qpWorkLength(m, n, useP, useA, allowPolish));
    }
}

/++
+/
@trusted pure nothrow @nogc
QPResult!(Workspace.Type) solveQP(Workspace)(ref scope Workspace workspace, bool polish = true, bool warmStart = false, QPSettings!(Workspace.Type) settings = QPSettings!(Workspace.Type).init)
    if (is(Workspace : QPWorkspace!(Workspace.Type)) || is(Workspace : RCQPWorkspace!(Workspace.Type)))
{
    import mir.rc.array;
    auto m = workspace.y.length;
    auto n = workspace.x.length;
    if (!warmStart)
    {
        workspace.x[][] = 0;
        workspace.y[][] = 0;
    }
    with(workspace)
        return solveQpImpl(
            m,
            n,
            P.ptr,
            q.ptr,
            A.ptr,
            l.ptr,
            u.ptr,
            x.ptr,
            y.ptr,
            z.ptr,
            work.length,
            work.ptr,
            iwork.ptr,
            polish,
            settings,
        );
}

/++
+/
@safe pure nothrow @nogc
size_t qpWorkLength(size_t m, size_t n, bool useP = true, bool useA = true, bool polish = true)
{
    if (!useP && !useA)
        return 0;
    auto workLen = n * (n + 2) + max(n * 64, n + m * 2 + n * 4 + max(n, m) * 2);
    if (useA && polish)
        workLen = workLen.max((n + m) * ((n + m) + (32 + 2) + 2)); // for KKT
    return workLen;
}

/++
+/
@system pure nothrow @nogc
QPResult!T solveQpImpl(T)(
    size_t m,
    size_t n,
    scope const(T)* P_ptr, // n x n (optional)
    scope const(T)* q_ptr, // n
    scope const(T)* A_ptr, // m x n (optional)
    scope const(T)* l_ptr, // n
    scope const(T)* u_ptr, // n
    scope T* x_ptr, // n
    scope T* y_ptr, // m
    scope T* z_ptr, // m
    size_t work_length, //
    scope T* work_ptr,
    scope lapackint* iwork_ptr, // n
    bool polish,
    ref scope const QPSettings!T settings)
{
    pragma(inline, false);
    bool useA() { return A_ptr !is null; }
    bool useP() { return P_ptr !is null; }

    auto A = A_ptr ? A_ptr.sliced(m, n) : A_ptr.sliced(0, 0);
    auto P = P_ptr ? P_ptr.sliced(n, n) : P_ptr.sliced(0, 0);
    auto q = q_ptr.sliced(n);
    auto l = l_ptr.sliced(n);
    auto u = u_ptr.sliced(n);
    auto x = x_ptr.sliced(n);
    auto y = y_ptr.sliced(m);
    auto z = z_ptr.sliced(m);

    void projection(Slice!(const(T)*) vec, Slice!(T*) result)
    {
        result[] = vec.zip(l).map!fmax.zip(u).map!fmin;
    }

    // trivial
    if (!useA && !useP)
    {
        projection(q, x);
        x[] = -x;
        QPResult!T ret = {
            criteria : QPTerminationCriteria.solved,
            objectiveValue : 0.5f * dot!T(x, x) + dot!T(x, q),
        };
        z[] = x; 
        y[] = -(x + q);
        return ret;
    }

    assert(work_length >= qpWorkLength(m, n, useP, useA, polish));

    auto work = work_ptr.sliced(work_length);
    auto iwork = iwork_ptr.sliced(n);

    assert(iwork.length == n);
    assert(work.length >= qpWorkLength(m, n, useP, useA, polish));

    QPResult!T ret = {criteria : QPTerminationCriteria.maxIterations};
    // approximate solution
    {
        auto buffer = work;
        auto F = buffer[0 .. n ^^ 2].sliced(n, n); buffer.popFrontExactly(n ^^ 2);
        auto F_e = buffer[0 .. n]; buffer.popFrontExactly(n);
        auto Ax = buffer[0 .. m]; buffer.popFrontExactly(m);

        auto qMax = T(0).reduce!fmax(map!fabs(q));

        if (useA)
        {
            gemv(1, A, x, 0, z);
            Ax[] = z;
        }
        else
        {
            z[] = x;
            Ax[] = x;
        }

        int rhoAge;
        if (!ret.rho) ret.rho = settings.initialRho;
        bool needUpdateRho = true;
        with(settings) for (; ret.iterations < maxIterations; ret.iterations++)
        {
            if (needUpdateRho)
            {
                rhoAge = 0;
                if (useP)
                    F[] = P;
                else
                    F[] = 0;
                with(settings)
                    F.diagonal[] += useA ? sigma : sigma + ret.rho;
                if (useA)
                    mir.blas.syrk!T(Uplo.Lower, ret.rho, A.transposed, 1, F);
                sytrf_rk!T('U', F.canonical, F_e, iwork, buffer);
                needUpdateRho = false;
            }

            auto xPrev = buffer[0 .. n];
            auto yPrev = buffer[n .. n + m];
            auto AxPrev = buffer[n + m .. n + m * 2];

            rhoAge++;
            xPrev[] = x;
            yPrev[] = y;
            y[] -= ret.rho * z;
            if (useA)
                gemv!T(-1, A.transposed, y, sigma, x);
            else
                x[] = sigma * x - y;
            x[] -= q;
            if (useA || useP)
                sytrs_3!T('U', F.canonical, F_e, iwork, x.sliced(1, x.length).canonical);
            else
                x[] *= 1 / (1 + sigma + ret.rho);
            x[] *= alpha;
            AxPrev[] = Ax;
            if (useA)
                gemv!T(1, A, x, 0, Ax);
            else
                Ax[] = x;
            x[] += (1 - alpha) * xPrev;
            y[] = Ax + (1 - alpha) * z + 1 / ret.rho * yPrev;
            projection(y, z);
            y[] = ret.rho * (y - z);
            Ax[] += (1 - alpha) * AxPrev;

            if ((ret.iterations + 1) % 6 && ret.iterations + 1 != maxIterations)
                continue;

            auto ATyATyPrev = buffer[n + m * 2 + n * 0 .. n + m * 2 + n * 2].sliced(2, n);
            auto ATy = ATyATyPrev[0];
            auto ATyPrev = ATyATyPrev[1];

            auto PxPxPrev = buffer[n + m * 2 + n * 2 .. n + m * 2 + n * 4].sliced(2, n);
            auto Px = PxPxPrev[0];
            auto PxPrev = PxPxPrev[1];

            if (useA)
            {
                auto yyPrev = buffer[n + m * 2 + n * 4 .. n + m * 2 + n * 4 + m * 2].sliced(2, n);
                yyPrev[0][] = y;
                yyPrev[1][] = yPrev;
                gemm!T(1, A.transposed, yyPrev.transposed, 0, ATyATyPrev.transposed);
            }
            else
            {
                ATy[] = y;
            }
            if (useP)
            {
                if (useA)
                {
                    auto xxPrev = buffer[n + m * 2 + n * 4 .. n + m * 2 + n * 4 + n * 2].sliced(2, n);
                    xxPrev[0][] = x;
                    xxPrev[1][] = xPrev;
                    symm!T(Side.Right, Uplo.Lower, 1, P, xxPrev, 0, PxPxPrev);
                }
                else
                {
                    symv!T(Uplo.Lower, 1, P, x, 0, Px);
                }
            }
            else
            {
                Px[] = x;
                if (useA)
                    PxPrev[] = xPrev;
            }
            T AxMax = T(0).reduce!fmax(map!fabs(Ax));
            T PxMax = T(0).reduce!fmax(map!fabs(Px));
            T zMax = T(0).reduce!fmax(map!fabs(z));
            T ATyMax = T(0).reduce!fmax(map!fabs(ATy));
            T primResidual = T(0).reduce!fmax(map!fabs(Ax - z));
            T primScale = fmax(AxMax, zMax);
            T dualResidual = T(0).reduce!fmax(map!fabs(Px + ATy + q));
            T dualScale = fmax(qMax, fmax(PxMax, ATyMax));

            T epsPrim = epsAbs + epsRel * primScale;
            T epsDual = epsAbs + epsRel * dualScale;
            if (primResidual <= epsPrim && dualResidual <= epsDual)
            {
                ret.criteria = QPTerminationCriteria.solved;
                if (!polish)
                {
                    Px[] = 0.5f * Px[] + q;
                    ret.objectiveValue = dot!T(x, Px);
                }
                break;
            }

            if (useA)
            {
                // check primal infeasibility
                yPrev[] -= y;
                T dyMax = T(0).reduce!fmax(map!fabs(yPrev));
                T dATyMax = T(0).reduce!fmax(map!fabs(ATy - ATyPrev));
                T epsPrimInfeasibility = epsInfeasibility * dyMax;

                if (dATyMax <= epsPrimInfeasibility)
                {
                    if (map!((d, ref u, ref l) => d * (d > 0 ? fmin(u, T.max.sqrt) : fmax(l, -T.max.sqrt)))
                        (zip(yPrev, u, l)).sum!"fast" <= epsPrimInfeasibility)
                        ret.criteria = QPTerminationCriteria.primalInfeasibility;
                        return ret;
                }

                xPrev[] -= x;
                T dxMax = T(0).reduce!fmax(map!fabs(xPrev));
                T dPxMax = T(0).reduce!fmax(map!fabs(Px - PxPrev));
                T epsDualInfeasibility = epsInfeasibility * dxMax;

                if (dPxMax <= epsDualInfeasibility && dot!T(q, xPrev) <= epsDualInfeasibility)
                {
                    foreach(i; 0 .. A.length)
                    {
                        auto dAxi = Ax[i] - AxPrev[i];
                        if (u[i] != +T.infinity)
                            if ((dAxi <= +epsDualInfeasibility))
                                goto F;
                        if (l[i] != -T.infinity)
                            if ((dAxi >= -epsDualInfeasibility))
                                goto F;
                    }
                    ret.criteria = QPTerminationCriteria.dualInfeasibility;
                    return ret;
                }
                F:
            }

            if (ret.rhoUpdates < maxRhoUpdates && rhoAge >= minAdaptiveRhoAge)
            {
                T primResidualNormalized = primResidual ? primResidual / primScale : 0;
                T dualResidualNormalized = dualResidual ? dualResidual / dualScale : 0;
                T newRho = ret.rho * sqrt(primResidualNormalized / dualResidualNormalized);
                if (fmax(ret.rho, newRho) > 5 * fmin(ret.rho, newRho))
                {
                    z[] = x;
                    ret.rho = newRho;
                    ret.rhoUpdates++;
                    needUpdateRho = true;
                }
            }
        }
    }
    // polish solution
    with (settings) if (polish)
    {
        size_t nonactiveLen; // first part of iwork
        size_t activeLen; // second part of iwork
        foreach (i; 0 .. cast(lapackint)n)
        {
            auto la = l[i] - z[i] >= y[i];
            auto ua = u[i] - z[i] <= y[i];
            y[i] = la && ua ? 0.5f * (l[i] + u[i]) : la ? l[i] : ua ? u[i] : 0;
            if (la || ua)
                iwork[$ - ++activeLen] = i;
            else
                iwork[nonactiveLen++] = i;
        }
        iwork[nonactiveLen .. $].reverseInPlace;
        auto nonactive = iwork[0 .. nonactiveLen];
        auto active = iwork[nonactiveLen .. $];
        if (useA)
        {
            auto kktn = n + activeLen;
            auto buffer = work;
            auto xy = buffer[0 .. kktn]; buffer.popFrontExactly(kktn); 
            auto KKT_e = buffer[0 .. kktn]; buffer.popFrontExactly(kktn); 
            auto KKT = buffer[0 .. kktn * kktn].sliced(kktn, kktn); buffer.popFrontExactly(kktn * kktn);
            auto KKT_work = buffer;
            KKT[0 .. n, 0 .. n] = P;
            assert(active.length == activeLen);
            foreach (i; 0 .. activeLen)
                KKT[n + i][0 .. n] = A[active[i]];
            KKT[n .. $, n .. $] = 0;
            xy[0 .. n] = -q;
            xy[n .. $] = y[active];
            syev('V', 'U', KKT.canonical, KKT_e, KKT_work);
            eigenSolve(KKT.canonical, KKT_e, xy, KKT_work, xy);
            x[] = xy[0 .. n];
            gemv(1, A, x, 0, z);
            y[active] = xy[n .. $];
        }
        // fast path
        else
        {
            auto buffer = work;
            auto P_ = buffer[0 .. nonactiveLen ^^ 2].sliced(nonactiveLen, nonactiveLen); buffer.popFrontExactly(nonactiveLen ^^ 2);
            auto G = buffer[0 .. nonactiveLen * activeLen].sliced(nonactiveLen, activeLen); buffer.popFrontExactly(nonactiveLen * activeLen);
            copyMinor(P, P_, nonactive, nonactive);
            copyMinor(P, G.transposed, active, nonactive);
            auto q_ = z[0 .. nonactiveLen];
            auto d = z[nonactiveLen .. $];
            q_[] = q[nonactive];
            auto e = buffer[0 .. nonactiveLen]; buffer.popFrontExactly(nonactiveLen);
            d[] = y[active];
            gemv!T(1, G, d, 1, q_);
            syev('V', 'L', P_.canonical, e, buffer);
            eigenSolve(P_.canonical, e, q_, buffer, q_, -1);
            x[] = y;
            x[nonactive] = q_;
            projection(x, x);
            z[] = x;
            y[] = q;
            symv!T(Uplo.Lower, -1, P, x, -1, y);
        }
        y[nonactive] = 0;
        auto Px_q = work[0 .. n];
        Px_q[] = q;
        gemv(0.5f, P, x, 1, Px_q);
        ret.objectiveValue = dot!T(x, Px_q);
    }
    return ret;
}

unittest
{
    import std.stdio;

    auto workspace = QPWorkspace!double(3, 3, true, false);
    workspace.P[] = [
        [ 2.0, -1, 0],
        [-1.0, 2, -1],
        [ 0.0, -1, 2],
    ];

    workspace.q[] = [3.0, -7, 5];
    workspace.l[] = [-100.0, -2, 1];
    workspace.u[] = [100.0, 2, 1];

    workspace.solveQP(false).writeln;
    import std.stdio;
    writeln(workspace.x);
    writeln(workspace.y);
    workspace.solveQP(true).writeln;
    writeln(workspace.x);
    writeln(workspace.y);
}

unittest
{
    import std.stdio;

    auto workspace = QPWorkspace!double(3, 3, true, true);
    workspace.P[] = [
        [ 2.0, -1, 0],
        [-1.0, 2, -1],
        [ 0.0, -1, 2],
    ];

    workspace.q[] = [3.0, -7, 5];
    workspace.l[] = [-100.0, -2, 1];
    workspace.u[] = [100.0, 2, 1];

    workspace.solveQP(false).writeln;
    import std.stdio;
    writeln(workspace.x);
    writeln(workspace.y);
    workspace.solveQP(true).writeln;
    writeln(workspace.x);
    writeln(workspace.y);
}

unittest
{
    import std.stdio;

    auto workspace = RCQPWorkspace!double(3, 3, true, false);
    workspace.P[][] = [
         2.0, -1, 0,
        -1.0, 2, -1,
         0.0, -1, 2,
    ];

    workspace.q[][] = [3.0, -7, 5];
    workspace.l[][] = [-100.0, -2, 1];
    workspace.u[][] = [100.0, 2, 1];

    workspace.solveQP(false).writeln;
    import std.stdio;
    writeln(workspace.x[]);
    writeln(workspace.y[]);
    workspace.solveQP(true).writeln;
    writeln(workspace.x[]);
    writeln(workspace.y[]);
}

unittest
{
    import std.stdio;

    auto workspace = RCQPWorkspace!double(3, 3, true, true);
    workspace.P[][] = [
         2.0, -1, 0,
        -1.0, 2, -1,
         0.0, -1, 2,
    ];

    workspace.q[][] = [3.0, -7, 5];
    workspace.l[][] = [-100.0, -2, 1];
    workspace.u[][] = [100.0, 2, 1];

    workspace.solveQP(false).writeln;
    import std.stdio;
    writeln(workspace.x[]);
    writeln(workspace.y[]);
    workspace.solveQP(true).writeln;
    writeln(workspace.x[]);
    writeln(workspace.y[]);
}
