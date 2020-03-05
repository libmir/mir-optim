module mir.optim.qp;

import mir.ndslice;
static import mir.blas;
static import mir.lapack;
static import lapack;
static import cblas;
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

///
void symv(T,
    SliceKind kindA,
    SliceKind kindX,
    SliceKind kindY,
    )(
    Uplo uplo,
    T alpha,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 1, kindX) x,
    T beta,
    Slice!(T*, 1, kindY) y,
    )
{
    assert(a.length!0 == a.length!1);
    assert(a.length!1 == x.length);
    assert(a.length!0 == y.length);
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;
    cblas.symv(
        transA ? cblas.Order.ColMajor : cblas.Order.RowMajor,
        uplo,
        
        cast(cblas.blasint) x.length,

        alpha,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,

        x.iterator,
        cast(cblas.blasint) x._stride,

        beta,

        y.iterator,
        cast(cblas.blasint) y._stride,
    );
}

///
void symm(T,
    SliceKind kindA,
    SliceKind kindB,
    SliceKind kindC,
    )(
    Side side,
    Uplo uplo,
    T alpha,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 2, kindB) b,
    T beta,
    Slice!(T*, 2, kindC) c,
    )
{
    assert(a.length!1 == a.length!0);
    assert(a.length!1 == b.length!0);
    assert(a.length!0 == c.length!0);
    assert(c.length!1 == b.length!1);
    auto k = cast(cblas.blasint) a.length!1;

    static if (kindC == Universal)
    {
        if (c._stride!1 != 1)
        {
            assert(c._stride!0 == 1, "Matrix C must have a stride equal to 1.");
            .symm(
                side == cblas.Side.Left ? cblas.Side.Right : cblas.Side.Left,
                uplo == cblas.Uplo.Upper ? cblas.Uplo.Lower : cblas.Uplo.Upper,
                alpha,
                a.transposed,
                b.transposed,
                beta,
                c.transposed.assumeCanonical);
            return;
        }
        assert(c._stride!1 == 1, "Matrix C must have a stride equal to 1.");
    }
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;
    static if (kindB == Universal)
    {
        bool transB;
        if (b._stride!1 != 1)
        {
            b = b.transposed;
            transB = true;
        }
        assert(b._stride!1 == 1, "Matrix B must have a stride equal to 1.");
    }
    else
        enum transB = false;

    cblas.symm(
        cblas.Order.RowMajor,
        transA ? cblas.Transpose.Trans : cblas.Transpose.NoTrans,
        transB ? cblas.Transpose.Trans : cblas.Transpose.NoTrans,
        
        cast(cblas.blasint) c.length!0,
        cast(cblas.blasint) c.length!1, 
        cast(cblas.blasint) k,

        alpha,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,

        b.iterator,
        cast(cblas.blasint) b.matrixStride,

        beta,

        c.iterator,
        cast(cblas.blasint) c.matrixStride,
    );
}


private auto matrixStride(S)(S a)
 if (S.N == 2)
{
    assert(a._stride!1 == 1);
    return a._stride != 1 ? a._stride : a.length!1;
}

/++@safe pure nothrow @nogc+/
// QPTerminationCriteria solveQuickQP(
//     scope ref const QPSolverSettings settings,
//     Slice!(const(T)*, 2, Canonical) P_eigenVectors,
//     Slice!(const(T)*) P_eigenValues,
//     Slice!(const(T)*) q,
//     Slice!(const(T)*) l,
//     Slice!(const(T)*) u,
//     Slice!(T*) x,
//     Slice!(T*) work,
//     scope QPTerminationCriteria delegate(
//         Slice!(const(T)*) xScaled,
//         Slice!(const(T)*) xScaledPrev,
//         Slice!(const(T)*) yScaled,
//         Slice!(const(T)*) yScaledPrev,
//     ) /++@safe pure nothrow @nogc+/ infeasibilityTolerance = null
// ) 
// {

import std.stdio;

struct QPSolver
{
    import mir.math.common: sqrt, fmin, fmax;
    import mir.math.constant: GoldenRatio;

    QPSolverSettings settings;

    bool useA() @property { return A.ptr !is null; }
    bool useP() @property { return P.ptr !is null; }

    T rho = QPSolverSettings.init.initialRho;

    Slice!(T*, 2) A;  // n x m
    Slice!(T*, 2) P; // P n
    Slice!(T*) q; // n
    Slice!(T*) l; // n
    Slice!(T*) u; // n

    Slice!(T*, 2) F; // LDL^t factorization of (P + σI + ρA^tA)

    Slice!(lapackint*) F_ipiv; //n
    Slice!(T*) F_work; // n
    Slice!(T*) F_e; // n

    Slice!(T*) x; // n
    Slice!(T*) xPrev; // n
    Slice!(T*) ATy; // n
    Slice!(T*) ATyPrev; // n
    Slice!(T*) Px; // n
    Slice!(T*) PxPrev; // n

    Slice!(T*) y; // m
    Slice!(T*) yPrev; // m
    Slice!(T*) z; // m
    Slice!(T*) Ax; // m
    Slice!(T*) AxPrev; // m

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

        if (useA || useP)
        {
            F = [n, n].slice!T(0);
            F_ipiv = [n].slice!lapackint(0);
            F_work = [n * 40].slice!T(0); // TODO fix
            F_e = [n].slice!T(0);
 
            xPrev = [n].slice!T(0);
            ATy = [n].slice!T(0);
            if (useA) ATyPrev = [n].slice!T(0);
            Px = [n].slice!T(0);
            PxPrev = [n].slice!T(0);
            yPrev = [m].slice!T(0);
            z = [m].slice!T(0);
            Ax = [m].slice!T(0);
            AxPrev = [m].slice!T(0);
       }
    }

    void updateF()
    {
        if (!useA && !useP)
            return;
        if (useP)
            F[] = P;
        else
            F[] = 0;
        with(settings)
            F.diagonal[] += useA ? sigma : sigma + rho;
        if (useA)
            mir.blas.syrk!T(Uplo.Lower, rho, A.transposed, 1, F);
        sytrf_rk!T('U', F.canonical, F_e, F_ipiv, F_work);
    }

    void polish()
    {
        if (!useA && !useP)
        {
            x[] = (-q).zip(l).map!fmax.zip(u).map!fmin;
            return;
        }
        size_t nonactiveLen; // first part of F_ipiv
        size_t activeLen; // second part of F_ipiv
        foreach (i; 0 .. cast(lapackint)F_ipiv.length)
        {
            auto la = l[i] - z[i] >= y[i];
            auto ua = u[i] - z[i] <= y[i];
            y[i] = la && ua ? 0.5f * (l[i] + u[i]) : la ? l[i] : ua ? u[i] : 0;
            if (la || ua)
                F_ipiv[$ - ++activeLen] = i;
            else
                F_ipiv[nonactiveLen++] = i;
        }
        F_ipiv[nonactiveLen .. $].reverseInPlace;
        auto nonactive = F_ipiv[0 .. nonactiveLen];
        auto active = F_ipiv[nonactiveLen .. $];
        if (!useA)
        {
            auto P_ = F[0 .. nonactiveLen, 0 .. nonactiveLen];
            auto G = F[0 .. nonactiveLen, nonactiveLen .. $];
            copyMinor(P, P_, nonactive, nonactive);
            copyMinor(P, G.transposed, active, nonactive);
            auto q_ = PxPrev[0 .. nonactiveLen];
            auto d = PxPrev[nonactiveLen .. $];
            q_[] = q[nonactive];
            auto e = F_e[0 .. nonactiveLen];
            d[] = y[active];
            gemv!T(1, G, d, 1, q_);
            syev('V', 'L', P_.canonical, e, F_work);
            eigenSolve(P_.canonical, e, q_, F_work, q_, -1);
            x[] = y;
            x[nonactive] = q_;
            x[] = x.zip(l).map!fmax.zip(u).map!fmin;
            y[] = q;
            symv!T(Uplo.Lower, -1, P, x, -1, y);
        }
        else
        {
            auto n = F_ipiv.length;
            auto kktn = n + activeLen;
            auto KKT_e = slice!T(kktn);
            auto xy = slice!T(kktn);
            auto KKT = slice!T([kktn, kktn], 0);
            auto KKT_work = slice!T(kktn * 40); // TODO fix
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
            y[active] = xy[n .. $];
        }
        y[nonactive] = 0;
    }

    QPTerminationCriteria solve()
    {
        int rhoAge;
        rho = settings.initialRho;
        auto qMax = T(0).reduce!fmax(map!fabs(q));
        z[] = 0;
        updateF;
        with(settings) foreach (iters; 0 .. maxIterations)
        {
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
                sytrs_3!T('U', F.canonical, F_e, F_ipiv, x.sliced(1, x.length).canonical);
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


            if (useA)
            {
                ATyPrev[] = ATy;
                PxPrev[] = Px;
                // TODO: use single gemm call
                gemv!T(1, A.transposed, y, 0, ATy);
                gemv!T(1, A.transposed, yPrev, 0, ATyPrev);
            }
            else
            {
                ATy[] = y;
            }
            if (useP)
            {
                // TODO: use single symm call
                symv!T(Uplo.Lower, 1, P, x, 0, Px);
                if (useA)
                    symv!T(Uplo.Lower, 1, P, xPrev, 0, PxPrev);
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
                return QPTerminationCriteria.solved;

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
                      rhoAge = 0;
                    updateF();
                }
            }
        }
        return QPTerminationCriteria.maxIterations;
    }


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

    solver.solve;
    // solver.polish;
    import std.stdio;
    writeln(solver.x);
    writeln(solver.y);
    solver.polish;
    writeln(solver.x);
    writeln(solver.y);
}
