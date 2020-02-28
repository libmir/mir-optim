module mir.optim.qp;

import mir.ndslice;
static import mir.blas;
static import mir.lapack;
static import lapack;
static import cblas;
import lapack: lapackint;

import mir.utility: min, max;
import mir.math.common: fmin, fmax, fabs, sqrt;
import mir.math.constant: GoldenRatio;
import mir.blas: gemv;

alias T = double;

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
    T initialRho = T(1) / 8;
    /++
    +/
    T minRho = T.epsilon.sqrt;
    /++
    +/
    T maxRho = 1 / T.epsilon.sqrt; 
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

@safe pure nothrow @nogc
void divBy(Slice!(T*) target, Slice!(const(T)*) roots)
{
    foreach (i; 0 .. target.length)
        target[i] *= 1 / roots[i];
}

@safe pure nothrow @nogc
void mulBy(Slice!(T*) target, Slice!(const(T)*) roots)
{
    foreach (i; 0 .. target.length)
        target[i] *= roots[i];
}

// @safe pure nothrow @nogc
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
//     ) @safe pure nothrow @nogc infeasibilityTolerance = null
// ) 
// {

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

QPTerminationCriteria solveQuickQP(
    Slice!(const(T)*, 2, Canonical) P,
    Slice!(const(T)*) q,
    Slice!(const(T)*) l,
    Slice!(const(T)*) r,
    Slice!(T*) x,
    QPSolverSettings settings = QPSolverSettings.init,
)
{
    return QPTerminationCriteria.init;
}

QPTerminationCriteria solveTrivialQP(
    Slice!(const(T)*) q,
    Slice!(const(T)*) l,
    Slice!(const(T)*) r,
    Slice!(T*) x,
    QPSolverSettings settings = QPSolverSettings.init,
)
{

    return QPTerminationCriteria.init;
}

@safe pure nothrow @nogc
private void eigenSqrtTimes(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValuesRoots,
    Slice!(const(T)*) x,
    Slice!(T*) y,
)
in {
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenValuesRoots.length <= eigenVectors.length);
    assert(x.length == eigenVectors.length);
    assert(y.length == eigenValuesRoots.length);
}
body {
    auto r = eigenValuesRoots.length;
    gemv!T(1, eigenVectors[0 .. r], x, 0, y);
    y.mulBy(eigenValuesRoots);
}

@safe pure nothrow @nogc
private void eigenSqrtSolve(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValuesRoots,
    Slice!(T*) y,
    Slice!(T*) x,
)
in {
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenValuesRoots.length <= eigenVectors.length);
    assert(x.length == eigenVectors.length);
    assert(y.length == eigenValuesRoots.length);
}
body {
    auto r = eigenValuesRoots.length;
    y.divBy(eigenValuesRoots);
    gemv!T(1, eigenVectors[0 .. r].transposed, y, 0, x);
}

@safe pure nothrow @nogc
private void eigenSqrtSplit(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValuesRoots,
    Slice!(const(T)*) x,
    Slice!(T*) y,
)
in {
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenVectors.length!0 == eigenVectors.length!1);
    assert(eigenValuesRoots.length <= eigenVectors.length);
    assert(x.length == eigenVectors.length);
    assert(y.length == eigenValuesRoots.length);
}
body {
    auto r = eigenValuesRoots.length;
    gemv!T(1, eigenVectors[0 .. r], x, 0, y);
    y.divBy(eigenValuesRoots);
}

@safe pure nothrow @nogc
private void svdTimes(
    Slice!(const(T)*, 2, Canonical) leftSingularVectors,
    Slice!(const(T)*, 2, Canonical) rightSingularVectors,
    Slice!(const(T)*) singularValues,
    Slice!(const(T)*) x,
    Slice!(T*) temp,
    Slice!(T*) y,
)
in {
    assert(leftSingularVectors.length!0 == leftSingularVectors.length!1);
    assert(rightSingularVectors.length!0 == rightSingularVectors.length!1);
    assert(singularValues.length <= min(leftSingularVectors.length, rightSingularVectors.length));
    assert(singularValues.length <= temp.length);
    assert(x.length == rightSingularVectors.length);
    assert(y.length == leftSingularVectors.length);
}
body {
    auto r = singularValues.length;
    temp = temp[0 ..r];
    gemv!T(1, rightSingularVectors[0 .. r], x, 0, temp);
    temp.mulBy(singularValues);
    gemv!T(1, leftSingularVectors[0 .. r].transposed, temp, 0, y);
}

@safe pure nothrow @nogc
private void svdSolve(
    Slice!(const(T)*, 2, Canonical) leftSingularVectors,
    Slice!(const(T)*, 2, Canonical) rightSingularVectors,
    Slice!(const(T)*) singularValues,
    Slice!(const(T)*) y,
    Slice!(T*) temp,
    Slice!(T*) x,
)
in {
    assert(leftSingularVectors.length!0 == leftSingularVectors.length!1);
    assert(rightSingularVectors.length!0 == rightSingularVectors.length!1);
    assert(singularValues.length <= min(leftSingularVectors.length, rightSingularVectors.length));
    assert(singularValues.length <= temp.length);
    assert(x.length == rightSingularVectors.length);
    assert(y.length == leftSingularVectors.length);
}
body {
    auto r = singularValues.length;
    temp = temp[0 ..r];
    gemv!T(1, leftSingularVectors.transposed[0 .. r], y, 0, temp);
    temp.divBy(singularValues);
    gemv!T(1, rightSingularVectors[0 .. $, 0 .. r], temp, 0, x);
}

/++
+/
@safe pure nothrow @nogc
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
    ) @safe pure nothrow @nogc projection,
    scope QPTerminationCriteria delegate(
        Slice!(const(T)*) xScaled,
        Slice!(const(T)*) xScaledPrev,
        Slice!(const(T)*) yScaled,
        Slice!(const(T)*) yScaledPrev,
    ) @safe pure nothrow @nogc infeasibilityTolerance = null
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
@safe pure nothrow @nogc
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
    ) @safe pure nothrow @nogc projection,
    scope QPTerminationCriteria delegate(
        Slice!(const(T)*) xScaled,
        Slice!(const(T)*) xScaledPrev,
        Slice!(const(T)*) yScaled,
        Slice!(const(T)*) yScaledPrev,
    ) @safe pure nothrow @nogc infeasibilityTolerance = null
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
            eigenSqrtSolve(P_eigenVectors, eroots, innerZ, x);
            projection(x, z); // set Z
            eigenSqrtTimes(P_eigenVectors, eroots, z, innerZ);
        },
        infeasibilityTolerance
    );
    // set Y
    eigenSqrtSolve(P_eigenVectors, eroots, innerY, y);
    return ret;
}

/++
+/
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
    ) @safe pure nothrow @nogc projection,
    scope QPTerminationCriteria delegate(
        Slice!(const(T)*) x,
        Slice!(const(T)*) xPrev,
        Slice!(const(T)*) y,
        Slice!(const(T)*) yPrev,
    ) @safe pure nothrow @nogc infeasibilityTolerance = null
) @safe pure nothrow @nogc
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
    xPrev[] = x;
    z[] = x;
    yPrev[] = y;
    int rhoAge;
    T qMax = T(0).reduce!fmax(map!fabs(q));
    T rho = settings.initialRho;
    with(settings) foreach (i; 0 .. maxIterations)
    {
        rho = rho.fmin(maxRho).fmax(minRho);

        x[] = GoldenRatio * (1 / (1 + rho)) * (rho * z - yPrev - q);
        y[] = x + (1 - GoldenRatio) * z + 1 / rho * yPrev;
        x[] += (1 - GoldenRatio) * xPrev;
        projection(y, z);
        y[] = rho * (y - z);

        T xMax = T(0).reduce!fmax(map!fabs(x));
        T yMax = T(0).reduce!fmax(map!fabs(y));
        T zMax = T(0).reduce!fmax(map!fabs(z));
        T primResidual = T(0).reduce!fmax(map!fabs(x - z));
        T dualResidual = T(0).reduce!fmax(map!fabs(x + y + q));
        T primScale = fmax(xMax, zMax);
        T dualScale = fmax(qMax, fmax(xMax, yMax));
        T primResidualNormalized = primResidual ? primResidual / primScale : 0;
        T dualResidualNormalized = dualResidual ? dualResidual / dualScale : 0;

        T epsPrim = epsAbs + epsRel * primScale;
        T epsDual = epsAbs + epsRel * dualScale;
        if (primResidual <= epsPrim && dualResidual <= epsDual)
            return QPTerminationCriteria.solved;

        if (infeasibilityTolerance)
            if (auto criteria = infeasibilityTolerance(x, xPrev, y, yPrev))
                return criteria;

        if (adaptiveRho && ++rhoAge >= minAdaptiveRhoAge)
        {
            T newRho = rho * sqrt(primResidualNormalized / dualResidualNormalized);
            if (fmax(rho, newRho) > 5 * fmin(rho, newRho))
            {
                z[] = x;
                rho = newRho;
                rhoAge = 0;
            }
        }
    }
    return QPTerminationCriteria.maxIterations;
}

version(nono):

///
size_t sytrf_rk(T)(
    char uplo,
    Slice!(T*, 2, Canonical) a,
    Slice!(T*) e,
    Slice!(lapackint*) ipiv,
    Slice!(T*) work,
    )
in
{
    assert(a.length!0 == a.length!1, "sytrf_rk: 'a' must be a square matrix.");
    assert(e.length == a.length, "sytrf_rk: 'e' must have the same length as 'a'.");
}
do
{
    lapackint info = void;
    lapackint n = cast(lapackint) a.length;
    lapackint lda = cast(lapackint) a._stride.max(1);
    lapackint lwork = cast(lapackint) work.length;
    lapack.sytrf_rk_(uplo, n, a.iterator, lda, e.iterator, ipiv.iterator, work.iterator, lwork, info);
    assert(info >= 0);
    return info;
}

unittest
{
    alias s = sytrf_rk!float;
    alias d = sytrf_rk!double;
    alias c = sytrf_rk!cfloat;
    alias z = sytrf_rk!cdouble;
}


///
size_t sytrf_rk_wk(T)(
    char uplo,
    Slice!(T*, 2, Canonical) a,
    )
in
{
    assert(a.length!0 == a.length!1, "sytrf_rk_wk: 'a' must be a square matrix.");
}
do
{

    lapackint info = void;
    lapackint n = cast(lapackint) a.length;
    lapackint lda = cast(lapackint) a._stride.max(1);
    lapackint lwork = -1;
    lapackint info = void;
    T e = void;
    T work = void;
    lapackint ipiv = void;

    lapack.sytrf_rk_(uplo, n, a.iterator, lda, &e, &ipiv, &work, lwork, info);

    return cast(size_t) work;
}

unittest
{
    alias s = sytrf_rk!float;
    alias d = sytrf_rk!double;
}

///
size_t sytrf_wk(T)(
    char uplo,
    Slice!(T*, 2, Canonical) a,
    )
in
{
    assert(a.length!0 == a.length!1, "sytrf_wk: 'a' must be a square matrix.");
}
do
{

    lapackint info = void;
    lapackint n = cast(lapackint) a.length;
    lapackint lda = cast(lapackint) a._stride.max(1);
    lapackint lwork = -1;
    lapackint info = void;
    T work = void;
    lapackint ipiv = void;

    lapack.sytrf_(uplo, n, a.iterator, lda, &ipiv, &work, lwork, info);

    return cast(size_t) work;
}

unittest
{
    alias s = sytrf_wk!float;
    alias d = sytrf_wk!double;
}

///
size_t sytrs_3(T)(
    char uplo,
    Slice!(T*, 2, Canonical) a,
    Slice!(T*) e,
    Slice!(lapackint*) ipiv,
    Slice!(T*, 2, Canonical) b,
    Slice!(T*) work,
    )
in
{
    assert(a.length!0 == a.length!1, "sytrs_3: 'a' must be a square matrix.");
    assert(e.length == a.length, "sytrs_3: 'e' must have the same length as 'a'.");
    assert(b.length!1 == a.length, "sytrs_3: 'b.length!1' must must be equal to 'a.length'.");
}
do
{
    lapackint n = cast(lapackint) a.length;
    lapackint nrhs = cast(lapackint) b.length;
    lapackint lda = cast(lapackint) a._stride.max(1);
    lapackint ldb = cast(lapackint) b._stride.max(1);
    lapackint info = void;
    lapack.sytrs_3_(uplo, n, nrhs, a.iterator, lda, e.iterator, ipiv.iterator, b.iterator, ldb, work.iterator, info);
    assert(info >= 0);
    return info;
}

unittest
{
    alias s = sytrs_3!float;
    alias d = sytrs_3!double;
    alias c = sytrs_3!cfloat;
    alias z = sytrs_3!cdouble;
}

///
auto iamax(T,
    SliceKind kindX,
    )(
    Slice!(const(T)*, 1, kindX) x,
    )
{
    return cblas.iamax(
        cast(cblas.blasint) x.length,
        x.iterator,
        cast(cblas.blasint) x._stride,
    );
}

auto famaxElem(T, SliceKind kind)(Slice!(const(T)*, 1, kind) x)
{
    return reduce!fmax(-T.infinity, x.map!fabs);
}

auto infNormDistance(T)(Slice!(const(T)*) x, Slice!(const(T)*) y)
{
    return reduce!fmax(-T.infinity, map!fabs(x - y));
}

auto faminElem(T, SliceKind kind)(Slice!(const(T)*, 1, kind) x)
{
    return reduce!fmin(+T.infinity, x.map!fabs);
}

struct QPSolver(T)
    if (is(T == float) || is(T == double))
{
    import mir.math.common: sqrt, fmin, fmax;
    import mir.math.constant: GoldenRatio;

    T sigma = T.epsilon.sqrt;
    T rho = 1;
    T alpha = GoldenRatio;
    T epsRel = T.epsilon.sqrt.sqrt * 3;
    T epsAbs = T.epsilon.sqrt.sqrt * 3;
    T epsPrimalInfeasibility = T.epsilon.sqrt.sqrt;
    T epsDualInfeasibility = T.epsilon.sqrt.sqrt;

    size_t n;
    size_t m;

    Slice!(T, 1, Universal) sigmaI() @trusted @property
    {
        return typeof(return)(&sigma, [n], [sizediff_t(1)]);
    }

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
    Slice!(T*) x_prev; // n
    Slice!(T*, 4) buff_4n; // 2 * n

    Slice!(T*) y; // m
    Slice!(T*) y_prev; // m
    Slice!(T*) z; // m
    Slice!(T*) Ax; // m
    Slice!(T*) Ax_prev; // m
    Slice!(T*) buff_m; // m

    void updateF()
    {
        mir.blas.copy!T(P.flattened, F.flattened);
        mir.blas.axpy!T(1, sigmaI, F.diagonal);
        mir.blas.syrk!T(Uplo.lower, rho, A.transposed, 1, F);
        sytrf_rk!T('U', F, F_e, F_ipiv, F_work);
    }

    QPTerminationCriteria iterate()
    {
        mir.blas.copy!T(y, y_prev);
        mir.blas.scale!T(-1, y);
        mir.blas.axpy!T(rho, z, y);
        mir.blas.copy!T(x, x_prev);
        if (useA)
            mir.blas.gemv!T(1, A.transposed, y, 0, x);
        else
            mir.blas.copy!T(y, x);
        mir.blas.axpy!T(-1, q, x);
        mir.blas.axpy!T(sigma, x_prev, x);
        if (useA || useP)
            sytrs_3!T('U', F, F_e, F_ipiv, x(1, n));
        else
            mir.blas.scale(1 / (1 + sigma + rho), x);
        mir.blas.scale!T(alpha, x);
        mir.blas.scale!T(1 - alpha, z);
        mir.blas.copy!T(Ax, Ax_prev);
        mir.blas.scale!T(1 - alpha, Ax);
        if (useA)
            mir.blas.gemv!T(1, A, x, 0, y);
        else
            mir.blas.copy!T(x, y);
        mir.blas.axpy!T(1, y, z);
        mir.blas.axpy!T(1, y, Ax);
        mir.blas.axpy!T(1 - alpha, x_prev, x);
        mir.blas.copy!T(z, y);
        mir.blas.axpy!T(1 / rho, y_prev, z);
        foreach (i, ref e; z.field)
            e = e.fmax(l[i]).fmin(u[i]);
        mir.blas.axpy!T(-1, z, y);
        mir.blas.scale!T(rho, y);
        mir.blas.axpy!T(1, y_prev, y);

        mir.blas.copy!T(Ax, buff_4n[0]);
        mir.blas.axpy!T(-1, z, buff_4n[0]);
        if (famaxElem(buff_4n[0]) < epsAbs + epsRel * famaxElem(z).fmax(famaxElem(Ax)))
            return QPTerminationCriteria.primal;
        return QPTerminationCriteria.none;
    }

    QPTerminationCriteria criteria()
    {
        if (useA)
        {
            mir.blas.copy!T(x, buff_4n[1]);
            mir.blas.axpy!T(-1, x_prev, buff_4n[1]);
            if (useP)
                mir.blas.symm!T(Side.right, Uplo.lower, 1, P, buff_4n[0 .. 2], 0, buff_4n[2 .. 4]);
            else
                mir.blas.copy(buff_4n[0 .. 2].flattened, 0, buff_4n[2 .. 4].flattened);
            mir.blas.gemv!T(1, A.transposed, y, 0, buff_4n[0]);
        }
        else
        {
            if (useP)
                mir.blas.symv!T(Uplo.lower, 1, P, x, 0, buff_4n[2]);
            else
                mir.blas.copy(x, buff_4n[2]);
            mir.blas.copy!T(y, buff_4n[0]);
        }

        T prim = famaxElem(q)
            .fmax(famaxElem(buff_4n[2]))
            .fmax(famaxElem(buff_4n[0]));
        mir.blas.axpy(1, q, buff_4n[0]);
        mir.blas.axpy(1, buff_4n[2], buff_4n[0]);

        if (famaxElem(buff_4n[0]) < epsAbs + epsRel * prim)
            return QPTerminationCriteria.dual;

        if (useA)
        {
            T normPdx = famaxElem(buff_4n[2] - buff_4n[3]);
        }

        return QPTerminationCriteria.none;
    }


    // uses buff_n as new #x
    void update()
    {
    }
}
