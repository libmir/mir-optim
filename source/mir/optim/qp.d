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
    int minAdaptiveRhoAge = 4000;
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

struct QPSolver
{
    import mir.math.common: sqrt, fmin, fmax;
    import mir.math.constant: GoldenRatio;

    QPSolverSettings settings;

    bool useA;
    bool useP;

    T qMax;

    size_t n;
    size_t m;

    Slice!(T*, 2) A;  // n x m
    Slice!(T*, 2) P; // P n
    Slice!(T*, 2) F; // LDL^t factorization of (P + σI + ρA^tA)

    Slice!(T*) q; // n
    Slice!(T*) l; // n
    Slice!(T*) u; // n

    Slice!(lapackint*) F_ipiv;
    Slice!(T*) F_work;
    Slice!(T*) F_e; // n

    Slice!(T*) x; // n
    Slice!(T*) xPrev; // n
    Slice!(T*) ATy; // n
    Slice!(T*) ATyPrev; // n
    Slice!(T*) Px; // n
    Slice!(T*) PxPrev; // n
    Slice!(T*, 4) buff_4n; // 2 * n

    Slice!(T*) y; // m
    Slice!(T*) yPrev; // m
    Slice!(T*) z; // m
    Slice!(T*) Ax; // m
    Slice!(T*) AxPrev; // m
    Slice!(T*) buff_m; // m

    // void updateF()
    // {
    //     mir.blas.copy!T(P.flattened, F.flattened);
    //     mir.blas.axpy!T(1, sigma, F.diagonal);
    //     mir.blas.syrk!T(Uplo.lower, rho, A.transposed, 1, F);
    //     sytrf_rk!T('U', F, F_e, F_ipiv, F_work);
    // }

    QPTerminationCriteria iterate()
    {
        auto rho = settings.initialRho;
        with(settings) foreach (iters; 0 .. maxIterations)
        {
            xPrev[] = x;
            yPrev[] = y;
            y[] -= rho * z;
            if (useA)
                gemv!T(-1, A.transposed, y, sigma, x);
            else
                x[] = sigma * x - y;
            x[] -= q;
            sytrs_3!T('U', F.canonical, F_e, F_ipiv, x.sliced(x.length, 1).canonical);
            x[] *= alpha;
            AxPrev[] = Ax;
            if (useA)
                gemv!T(1, A, x, 0, Ax);
            else
                Ax[] = x;
            y[] = Ax + (1 - alpha) * z + 1 / rho * y;
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
            T primResidualNormalized = primResidual ? primResidual / primScale : 0;
            T dualResidual = T(0).reduce!fmax(map!fabs(Px + ATy + q));
            T dualScale = fmax(qMax, fmax(PxMax, ATyMax));
            T dualResidualNormalized = dualResidual ? dualResidual / dualScale : 0;

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
                    if (map!((d, ref u, ref l) => d * (d > 0 ? fmin(u, T.max.sqrt) : fmax(l, -T.max.sqrt)))(zip(yPrev, u, l)).sum!"fast" <= epsPrimInfeasibility)
                        return QPTerminationCriteria.primalInfeasibility;
                }

                xPrev[] -= x;
                T dxMax = T(0).reduce!fmax(map!fabs(xPrev));
                T dPxMax = T(0).reduce!fmax(map!fabs(Px - PxPrev));
                T epsDualInfeasibility = epsInfeasibility * dxMax;

                if (dPxMax <= epsDualInfeasibility && dot!T(q, xPrev) <= epsDualInfeasibility)
                {
                    foreach(i; 0 .. m)
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

            // if (adaptiveRho && ++rhoAge >= minAdaptiveRhoAge)
            // {
            //     T newRho = rho * sqrt(primResidualNormalized / dualResidualNormalized);
            //     if (fmax(rho, newRho) > 5 * fmin(rho, newRho))
            //     {
            //         z[] = x;
            //         rho = newRho;
            //           rhoAge = 0;
            //     }
            // }
        }
        return QPTerminationCriteria.maxIterations;
    }
}

version(none):

QPTerminationCriteria solveQP(
    Slice!(const(T)*, 2, Canonical) P,
    Slice!(const(T)*) q,
    Slice!(const(T)*, 2, Canonical) A,
    Slice!(const(T)*) l,
    Slice!(const(T)*) r,
    Slice!(T*) x,
    QPSolverSettings settings = QPSolverSettings.init,
)
{
    return QPTerminationCriteria.init;
}

// /++@safe pure nothrow @nogc+/
QPTerminationCriteria solveQuickQP(
    Slice!(const(T)*, 2, Canonical) P,
    Slice!(const(T)*) q,
    Slice!(const(T)*) l,
    Slice!(const(T)*) u,
    Slice!(T*) x,
    QPSolverSettings settings = QPSolverSettings.init,
)
in {
    assert(q.length == x.length);
    assert(l.length == x.length);
    assert(u.length == x.length);
    assert(P.length == x.length);
    assert(P.length!0 == P.length!1);
}
do {
    auto n = x.length;
    auto eigenWork = syev_wk('V', 'L', Slice!(T*, 2, Canonical)([n, n], [n], null), x);
    auto work = safeAlloc!T(n * n + n + max(eigenWork, n * 9)).sliced;
    auto vectors = work[0 .. n * n].sliced(n, n); work = work[n * n .. $];
    auto values = work[0 .. n]; work = work[n .. $];
    vectors[] = P;
    syev('V', 'L', vectors.canonical, values, work);
    auto ret = embededSolveQuickQP(vectors.canonical, values, q, l, u, x, work, settings);
    (()@trusted => vectors.ptr.free)();
    return ret;
}

unittest
{
    // auto P = [
    //      2.0, -1, 0,
    //     -1, 2, -1,
    //      0, -1, 2,
    // ].sliced(3, 3);

    auto P = [
         1.0, -0, 0,
        -0, 1, -0,
         0, -0, 1,
    ].sliced(3, 3);

    auto q = [3.0, -7, 5].sliced;
    auto l = [-100.0, -2, 1].sliced;
    auto u = [100.0, 100, 1].sliced;
    auto x = slice!double(3);
    solveQuickQP(P.canonical, q, l, u, x);
    writeln(x);
}

    import std.stdio;

///++@safe pure nothrow @nogc+/
QPTerminationCriteria embededSolveQuickQP(
    Slice!(const(T)*, 2, Canonical) P_eigenVectors,
    Slice!(const(T)*) P_eigenValues,
    Slice!(const(T)*) q,
    Slice!(const(T)*) l,
    Slice!(const(T)*) u,
    Slice!(T*) x,
    Slice!(T*) work,
    QPSolverSettings settings = QPSolverSettings.init,
)
in {
    assert(q.length == x.length);
    assert(l.length == x.length);
    assert(u.length == x.length);
    assert(work.length >= x.length * 9);
    assert(P_eigenValues.length == x.length);
    assert(P_eigenVectors.length!0 == P_eigenVectors.length!1);
    assert(P_eigenValues[0] <= T.max);
}
do {
    auto n = x.length;
    auto r = n;
    while (r && P_eigenValues[r - 1] < T.min_normal) r--;
    P_eigenValues = P_eigenValues[0 .. r];
    auto y = work[0 .. n]; work = work[n .. $];
    auto z = work[0 .. n]; work = work[n .. $];
    eigenSolve(P_eigenVectors, P_eigenValues, q, work, z, -1);
    projection(l, u, z, x);
    // if (z == x)
    //     return QPTerminationCriteria.solved;
    x[] = 0;
    y[] = 0;
    auto ret = approxSolveQuickQP(settings,
        P_eigenVectors, P_eigenValues,
        q, x, y, z, work,
        (x, z) => projection(l, u, x, z)
    );
    debug writeln("approx x: ", x);
    debug writeln("approx y: ", y);
    foreach (i; 0 .. n)
    {
        auto la = l[i] - z[i] > y[i];
        auto ua = u[i] - z[i] < y[i];
        x[i] = la && ua ? 0.5f * (l[i] + u[i]) : la ? l[i] : ua ? u[i] : 0;
    }
    eigenSolve(P_eigenVectors, P_eigenValues, x, work, x);
    foreach (i; 0 .. n)
    {
        auto la = l[i] - z[i] > y[i];
        auto ua = u[i] - z[i] < y[i];
        if (!la && !ua)
            x[i] = -q[i];
    }
    eigenSolve(P_eigenVectors, P_eigenValues, x, work, x);
    projection(l, u, x, x);
    return ret;
}

// /++@safe pure nothrow @nogc+/
QPTerminationCriteria solveTrivialQP(
    Slice!(const(T)*) q,
    Slice!(const(T)*) l,
    Slice!(const(T)*) u,
    Slice!(T*) x,
    QPSolverSettings settings = QPSolverSettings.init,
)
in {
    assert(q.length == l.length);
    assert(q.length == u.length);
    assert(q.length == x.length);
}
do {
    x[] = -q;
    projection(l, u, x, x);
    return QPTerminationCriteria.solved;
}

/++
+/
// /++@safe pure nothrow @nogc+/
QPTerminationCriteria approxSolveQP(
    scope ref const QPSolverSettings settings,
    Slice!(const(T)*, 2, Canonical) A_leftSingularVectors,
    Slice!(const(T)*, 2, Canonical) A_rightSingularVectors,
    Slice!(const(T)*) A_singularValues,
    Slice!(const(T)*, 2, Canonical) P_eigenVectors,
    Slice!(const(T)*) P_eigenValues,
    Slice!(const(T)*) q,
    Slice!(T*) x,
    Slice!(T*) y,
    Slice!(T*) z,
    Slice!(T*) work,
    scope void delegate(
        Slice!(const(T)*) x,
        Slice!(T*) z,
    ) /++@safe pure nothrow @nogc+/ projection,
    scope QPTerminationCriteria delegate(
        Slice!(const(T)*) xScaled,
        Slice!(const(T)*) xScaledPrev,
        Slice!(const(T)*) yScaled,
        Slice!(const(T)*) yScaledPrev,
    ) /++@safe pure \nothrow @nogc+/ infeasibilityTolerance = null
) 
in {
    assert(projection);
    assert(x.length == y.length);
    assert(z.length == y.length);
    assert(work.length >= x.length * 8 + y.length * 2);
    assert(P_eigenValues.length == x.length);
    assert(P_eigenVectors.length!0 == P_eigenVectors.length!1);
    assert(P_eigenValues[0] <= T.max);

    assert(A_leftSingularVectors.length!0 == A_leftSingularVectors.length!1);
    assert(A_rightSingularVectors.length!0 == A_rightSingularVectors.length!1);
    assert(A_singularValues.length == min(A_leftSingularVectors.length, A_rightSingularVectors.length));

    assert(A_rightSingularVectors.length == x.length);
    assert(A_leftSingularVectors.length == z.length);
}
do {
    auto n = x.length;
    auto m = y.length;
    auto r = A_singularValues.length;
    while (r && A_singularValues[r - 1] < T.min_normal) r--;
    A_singularValues = A_singularValues[0 .. r];
    auto innerY = work[0 .. m]; work = work[m .. $];
    auto innerZ = work[0 .. m]; work = work[m .. $];
    auto tempN = work[0 .. n]; work = work[n .. $];
    svdSolve(A_leftSingularVectors, A_rightSingularVectors, A_singularValues, y, tempN, innerY);
    auto ret = approxSolveQuickQP(
        settings,
        P_eigenVectors,
        P_eigenValues,
        q,
        x,
        innerY,
        innerZ,
        work,
        // inner task
        (
            Slice!(const(T)*) innerX,
            Slice!(T*) innerZ,
        ){
            assert (innerX.length == n);
            assert (innerZ.length == n);
            svdTimes(A_leftSingularVectors, A_rightSingularVectors, A_singularValues, innerX, innerZ, tempN);
            projection(tempN, z); // set Z
            svdSolve(A_leftSingularVectors, A_rightSingularVectors, A_singularValues, z, tempN, innerZ);
        },
        infeasibilityTolerance
    );
    // set Y
    svdTimes(A_leftSingularVectors, A_rightSingularVectors, A_singularValues, innerY, tempN, y);
    return ret;
}

/++
+/
// /++@safe pure nothrow @nogc+/
QPTerminationCriteria approxSolveQuickQP(
    scope ref const QPSolverSettings settings,
    Slice!(const(T)*, 2, Canonical) P_eigenVectors,
    Slice!(const(T)*) P_eigenValues,
    Slice!(const(T)*) q,
    Slice!(T*) x,
    Slice!(T*) y,
    Slice!(T*) z,
    Slice!(T*) work,
    scope void delegate(
        Slice!(const(T)*) x,
        Slice!(T*) z,
    ) /++@safe pure nothrow @nogc+/ projection,
    scope QPTerminationCriteria delegate(
        Slice!(const(T)*) xScaled,
        Slice!(const(T)*) xScaledPrev,
        Slice!(const(T)*) yScaled,
        Slice!(const(T)*) yScaledPrev,
    ) /++@safe pure nothrow @nogc+/ infeasibilityTolerance = null
) 
in {
    assert(projection);
    assert(x.length == y.length);
    assert(z.length == x.length);
    assert(work.length >= x.length * 7);
    assert(P_eigenValues.length == x.length);
    assert(P_eigenVectors.length!0 == P_eigenVectors.length!1);
    assert(P_eigenValues[0] <= T.max);
}
do {
    import mir.blas: gemv;

    auto n = x.length;
    auto r = n;
    while (r && P_eigenValues[r - 1] < T.min_normal) r--;
    auto eroots = work[0 .. r]; work = work[r .. $];
    auto innerQ = work[0 .. r]; work = work[r .. $];
    auto innerX = work[0 .. r]; work = work[r .. $];
    auto innerY = work[0 .. r]; work = work[r .. $];
    auto innerZ = work[0 .. r]; work = work[r .. $];
    foreach (i; 0 .. r)
        eroots[i] = P_eigenValues[i].sqrt;
    eigenSqrtSplit(P_eigenVectors, eroots, q, innerQ);
    eigenSqrtTimes(P_eigenVectors, eroots, x, innerX);
    eigenSqrtTimes(P_eigenVectors, eroots, y, innerY);
    auto ret = approxSolveTrivialQP(
        settings,
        innerQ,
        innerX,
        innerY,
        innerZ,
        work,
        // inner task
        (
            Slice!(const(T)*) innerX,
            Slice!(T*) innerZ,
        ){
            assert (innerX.length == r);
            assert (innerZ.length == r);
            // use innerZ as temporal storage
            innerZ[] = innerX;
    // debug writeln("approxSolveQuickQP#.innerX = ", innerX);
    debug writeln("approxSolveQuickQP#.innerZ0 = ", innerZ);
            eigenSqrtSolve(P_eigenVectors, eroots, innerZ, x);
    debug writeln("approxSolveQuickQP#.innerZ1 = ", innerZ);
    debug writeln("approxSolveQuickQP#.x = ", x);
            projection(x, z); // set Z
            eigenSqrtTimes(P_eigenVectors, eroots, z, innerZ);
        },
        infeasibilityTolerance
    );
    // set Y
    debug writeln("approxSolveQuickQP.P_eigenVectors = ", P_eigenVectors);
    debug writeln("approxSolveQuickQP.P_eigenValues = ", P_eigenValues);
    debug writeln("approxSolveQuickQP.eroots = ", eroots);
    debug writeln("approxSolveQuickQP.innerX = ", innerX);
    debug writeln("approxSolveQuickQP.innerY = ", innerY);
    debug writeln("approxSolveQuickQP.q = ", q);
    debug writeln("approxSolveQuickQP.innerQ = ", innerQ);
    eigenSqrtSplitReverse(P_eigenVectors, eroots, innerQ, work[0 .. n]);
    eigenSqrtSolve(P_eigenVectors, eroots, innerX, x);
    eigenSqrtSolve(P_eigenVectors, eroots, innerY, y);
    eigenSqrtSolve(P_eigenVectors, eroots, innerZ, z);
    debug writeln("approxSolveQuickQP.q' = ", work[0 .. n]);
    debug writeln("approxSolveQuickQP.x = ", x);
    debug writeln("approxSolveQuickQP.y = ", y);
    return ret;
}

/++
+/
// /++@safe pure nothrow @nogc+/
QPTerminationCriteria approxSolveTrivialQP(
    scope ref const QPSolverSettings settings,
    Slice!(const(T)*) q,
    Slice!(T*) x,
    Slice!(T*) y,

    Slice!(T*) z,
    Slice!(T*) work,
    scope void delegate(
        Slice!(const(T)*) x,
        Slice!(T*) z,
    ) /++@safe pure nothrow @nogc+/ projection,
    scope QPTerminationCriteria delegate(
        Slice!(const(T)*) x,
        Slice!(const(T)*) xPrev,
        Slice!(const(T)*) y,
        Slice!(const(T)*) yPrev,
    ) /++@safe pure nothrow @nogc+/ infeasibilityTolerance = null
)
in {
    assert(projection);
    assert(x.length == y.length);
    assert(z.length == y.length);
    assert(work.length >= y.length * 2);
}
do {
    auto n = x.length;
    auto xPrev = work[n * 0 .. n * 1];
    auto yPrev = work[n * 1 .. n * 2];
    x[] = 0;
    y[] = 0;
    z[] = x;
    auto sigma = 1e-2;
    int rhoAge;
    T qMax = T(0).reduce!fmax(map!fabs(q));
    T rho = settings.initialRho;
    with(settings) foreach (i; 0 .. maxIterations)
    {
        rho = rho.fmin(maxRho).fmax(minRho);
        xPrev[] = x;
        yPrev[] = y;
        x *= sigma;
        x[] = 1.00000 * (1 / (1 + sigma + rho)) * (sigma * x + rho * z - yPrev - q);
        y[] = x + (1 - 1.00000) * z + 1 / rho * yPrev;
        x[] += (1 - 1.00000) * xPrev;
        // debug writeln(i, " y' = ", y);
        projection(y, z);
        y[] = rho * (y - z);
        debug writeln(i, " q = ", q);
        debug writeln(i, " x = ", x);
        debug writeln(i, " y = ", y);
        debug writeln(i, " z = ", z);
        debug writeln(i, " rho = ", rho);
        auto v = [-1.27373082,  1.50000355, -2.15119364].sliced;
        auto w = new T[n].sliced;
        projection(v, w);
        debug writeln(i, " w = ", w);
        debug writeln(i, " rw = ", T(0).reduce!fmax(map!fabs(v - w)));
            


        T xMax = T(0).reduce!fmax(map!fabs(x));
        T yMax = T(0).reduce!fmax(map!fabs(y));
        T zMax = T(0).reduce!fmax(map!fabs(z));
        T primResidual = T(0).reduce!fmax(map!fabs(x - z));
        T dualResidual = T(0).reduce!fmax(map!fabs(x + y + q));
        T primScale = fmax(xMax, zMax);
        T dualScale = fmax(qMax, fmax(xMax, yMax));
        T primResidualNormalized = primResidual ? primResidual / primScale : 0;
        T dualResidualNormalized = dualResidual ? dualResidual / dualScale : 0;

        debug writeln(i, " prim = ", primResidual);
        debug writeln(i, " dual = ", dualResidual);

        T epsPrim = epsAbs + epsRel * primScale;
        T epsDual = epsAbs + epsRel * dualScale;
        if (primResidual <= epsPrim && dualResidual <= epsDual)
            return QPTerminationCriteria.solved;

        if (infeasibilityTolerance)
            if (auto criteria = infeasibilityTolerance(x, xPrev, y, yPrev))
                return criteria;

        // if (adaptiveRho && ++rhoAge >= minAdaptiveRhoAge)
        // {
        //     T newRho = rho * sqrt(primResidualNormalized / dualResidualNormalized);
        //     if (fmax(rho, newRho) > 5 * fmin(rho, newRho))
        //     {
        //         z[] = x;
        //         rho = newRho;
        //         rhoAge = 0;
        //     }
        // }
    }
    return QPTerminationCriteria.maxIterations;
}

/++@safe pure nothrow @nogc+/
private void projection(Slice!(const(T)*) l, Slice!(const(T)*) u, Slice!(const(T)*) x, Slice!(T*) z)
in {
    assert(x.length == z.length);
    assert(l.length == z.length);
    assert(u.length == z.length);
}
do {
    foreach (i; 0 .. x.length)
        z[i] = x[i].fmax(l[i]).fmin(u[i]);
}
