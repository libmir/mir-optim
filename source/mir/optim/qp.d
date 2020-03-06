module mir.optim.qp;

import mir.ndslice;
static import mir.blas;
static import mir.lapack;
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
struct QPSolverSettings
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
    T initialRho = T(1) / 8;
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
    bool adaptiveRho = true;
    /++
    +/
    int minAdaptiveRhoAge = 10;
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

import std.stdio;

struct QPSolver
{
    import mir.math.common: sqrt, fmin, fmax;

    QPSolverSettings settings;
    T rho = 0;

    private @safe pure nothrow @nogc const @property
    {
        bool useA() { return A.ptr !is null; }
        bool useP() { return P.ptr !is null; }
    }

    Slice!(T*, 2) A;  // m x n
    Slice!(T*, 2) P; // n x n
    Slice!(T*) q; // n
    Slice!(T*) l; // n
    Slice!(T*) u; // n
    Slice!(T*) x; // n
    Slice!(T*) y; // m
    Slice!(T*) z; // m

    // Slice!(T*, 2) F; // n x n; LDL^t factorization of (P + σI + ρA^tA)
    // Slice!(lapackint*) iwork; //n
    // Slice!(T*) F_work; // n
    // Slice!(T*) F_e; // n

    // Slice!(T*) xPrev; // n
    // Slice!(T*) ATy; // n
    // Slice!(T*) ATyPrev; // n
    // Slice!(T*) Px; // n
    // Slice!(T*) PxPrev; // n

    // Slice!(T*) yPrev; // m
    // Slice!(T*) z; // m
    // Slice!(T*) Ax; // m
    // Slice!(T*) AxPrev; // m

    this(size_t n, size_t m, bool useP, bool useA)
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
    }

    // // @safe pure nothrow @nogc
    // void polish()
    // {
    //     auto n = x.length;
    //     auto m = y.length;
    // }

    // @safe pure nothrow @nogc
    QPTerminationCriteria solve(bool polish = true)
    {
        // trivial
        if (!useA && !useP)
        {
            x[] = q.zip(l).map!fmax.zip(u).map!fmin;
            x[] = -x;
            z[] = x;
            y[] = -(x + q);
            return QPTerminationCriteria.solved;
        }

        auto n = x.length;
        auto m = y.length;

        // compute defualt lapack and local workspace
        auto workLen = n * (n + 2) + max(n * 64, n + m * 2 + n * 4 + max(n, m) * 2);
        auto iworkLen = n;
        if (useA && polish)
        {
            iworkLen += m;
            workLen = workLen.max((n + m) * ((n + m) + (32 + 2) + 2)); // for KKT
            iworkLen += n;
        }

        auto work = [workLen].slice!T;
        auto iwork = [n].slice!lapackint;

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
            if (!rho) rho = settings.initialRho;
            bool needUpdateRho = true;
            with(settings) foreach (iters; 0 .. maxIterations)
            {
                if (needUpdateRho)
                {
                    rhoAge = 0;
                    if (useP)
                        F[] = P;
                    else
                        F[] = 0;
                    with(settings)
                        F.diagonal[] += useA ? sigma : sigma + rho;
                    if (useA)
                        mir.blas.syrk!T(Uplo.Lower, rho, A.transposed, 1, F);
                    sytrf_rk!T('U', F.canonical, F_e, iwork, buffer);
                    needUpdateRho = false;
                }

                auto xPrev = buffer[0 .. n];
                auto yPrev = buffer[n .. n + m];
                auto AxPrev = buffer[n + m .. n + m * 2];

                rhoAge++;
                xPrev[] = x;
                yPrev[] = y;
                y[] -= rho * z;
                if (useA)
                    gemv!T(-1, A.transposed, y, sigma, x);
                else
                    x[] = sigma * x - y;
                x[] -= q;
                if (useA || useP)
                    sytrs_3!T('U', F.canonical, F_e, iwork, x.sliced(1, x.length).canonical);
                else
                    x[] *= 1 / (1 + sigma + rho);
                x[] *= alpha;
                AxPrev[] = Ax;
                if (useA)
                    gemv!T(1, A, x, 0, Ax);
                else
                    Ax[] = x;
                x[] += (1 - alpha) * xPrev;
                y[] = Ax + (1 - alpha) * z + 1 / rho * yPrev;
                z[] = y.zip(l).map!fmax.zip(u).map!fmin;
                y[] = rho * (y - z);
                Ax[] += (1 - alpha) * AxPrev;

                if ((iters + 1) % 6 && iters + 1 != maxIterations)
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
                    break;

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
                            return QPTerminationCriteria.primalInfeasibility;
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
                        return QPTerminationCriteria.dualInfeasibility;
                    }
                    F:
                }

                if (adaptiveRho && rhoAge >= minAdaptiveRhoAge)
                {
                    T primResidualNormalized = primResidual ? primResidual / primScale : 0;
                    T dualResidualNormalized = dualResidual ? dualResidual / dualScale : 0;
                    T newRho = rho * sqrt(primResidualNormalized / dualResidualNormalized);
                    if (fmax(rho, newRho) > 5 * fmin(rho, newRho))
                    {
                        z[] = x;
                        rho = newRho;
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
            // fast path
            if (!useA)
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
                x[] = x.zip(l).map!fmax.zip(u).map!fmin;
                z[] = x;
                y[] = q;
                symv!T(Uplo.Lower, -1, P, x, -1, y);
            }
            else
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
            y[nonactive] = 0;

        }
        return QPTerminationCriteria.maxIterations;
    }


}

unittest
{
    import std.stdio;

    auto solver = QPSolver(3, 3, true, false);
    solver.P[] = [
        [ 2.0, -1, 0],
        [-1.0, 2, -1],
        [ 0.0, -1, 2],
    ];

    // solver.P[] = [
    //     [ 1.0,  0, 0],
    //     [ 0.0,  1, 0],
    //     [ 0.0,  0, 1],
    // ];

    solver.q[] = [3.0, -7, 5];
    solver.l[] = [-100.0, -2, 1];
    solver.u[] = [100.0, 2, 1];

    solver.solve(false);
    import std.stdio;
    writeln(solver.x);
    writeln(solver.y);
    solver.solve(true);
    writeln(solver.x);
    writeln(solver.y);
}

unittest
{
    import std.stdio;

    auto solver = QPSolver(3, 3, true, true);
    solver.P[] = [
        [ 2.0, -1, 0],
        [-1.0, 2, -1],
        [ 0.0, -1, 2],
    ];

    // solver.P[] = [
    //     [ 1.0,  0, 0],
    //     [ 0.0,  1, 0],
    //     [ 0.0,  0, 1],
    // ];

    solver.q[] = [3.0, -7, 5];
    solver.l[] = [-100.0, -2, 1];
    solver.u[] = [100.0, 2, 1];

    solver.solve(false);
    import std.stdio;
    writeln(solver.x);
    writeln(solver.y);
    solver.solve(true);
    writeln(solver.x);
    writeln(solver.y);
}
