module mir.optim.qp;

import mir.ndslice;
static import mir.blas;
static import mir.lapack;
static import lapack;
static import cblas;
import lapack: lapackint;

import mir.utility: max;
import mir.math.common: fmin, fmax, fabs, sqrt;
import mir.math.constant: GoldenRatio;

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
void divBy(scope T[] target, scope const T[] roots)
{
    foreach (i; 0 .. target.length)
        target[i] *= 1 / roots[i];
}

@safe pure nothrow @nogc
void mulBy(scope T[] target, scope const T[] roots)
{
    foreach (i; 0 .. target.length)
        target[i] *= roots[i];
}

/++
+/
@safe pure nothrow @nogc
QPTerminationCriteria approxSolveQP(
    scope ref const QPSolverSettings settings,
    Slice!(const(T)*, 2, Canonical) P_eigenVectors,
    scope const T[] P_eigenValues,
    scope const T[] q,
    scope T[] x,
    scope T[] y,
    scope T[] z,
    scope T[] work,
    scope void delegate(
        scope const(T)[] x,
        scope T[] z,
    ) @safe pure nothrow @nogc projection,
    scope QPTerminationCriteria delegate(
        scope const(T)[] xScaled,
        scope const(T)[] xScaledPrev,
        scope const(T)[] yScaled,
        scope const(T)[] yScaledPrev,
    ) @safe pure nothrow @nogc infeasibilityTolerance = null
) 
in {
    assert(projection);
    assert(x.length == y.length);
    assert(z.length == y.length);
    assert(work.length >= y.length * 5);
    assert(P_eigenValues.length == y.length);
    assert(P_eigenVectors.length!0 == y.length);
    assert(P_eigenVectors.length!1 == y.length);
    // assert(P_eigenValues[$ - 1] > T.min_normal);
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
    auto temp = work[0 .. n]; work = work[n .. $];
    foreach (i; 0 .. r)
        eroots[i] = P_eigenValues[i].sqrt;
    gemv!T(1, P_eigenVectors[0 .. r], q.sliced, 0, innerQ.sliced);
    innerQ.divBy(eroots);
    gemv!T(1, P_eigenVectors[0 .. $, 0 .. r], x.sliced, 0, innerX.sliced);
    innerX.mulBy(eroots);
    gemv!T(1, P_eigenVectors[0 .. $, 0 .. r], y.sliced, 0, innerY.sliced);
    innerY.mulBy(eroots);
    auto ret = approxSolveQP(
        settings,
        innerQ,
        innerX,
        innerY,
        innerZ,
        work,
        // inner task
        (
            scope const(T)[] innerX,
            scope T[] innerZ,
        ){
            assert (innerX.length == r);
            assert (innerZ.length == r);
            // use innerZ as temporal storage
            innerZ[] = innerX;
            innerZ.divBy(eroots);
            gemv!T(1, P_eigenVectors[0 .. r].transposed, innerZ.sliced, 0, x.sliced); // set X
            projection(x, z); // set Z
            gemv!T(1, P_eigenVectors[0 .. $, 0 .. r], z.sliced, 0, innerZ.sliced);
            innerZ.mulBy(eroots);
        },
        infeasibilityTolerance
    );
    // set Y
    innerY.divBy(eroots);
    gemv!T(1, P_eigenVectors[0 .. r].transposed, innerY.sliced, 0, y.sliced);
    return ret;
}

/++
+/
QPTerminationCriteria approxSolveQP(
    scope ref const QPSolverSettings settings,
    scope const T[] q,
    scope T[] x,
    scope T[] y,
    scope T[] z,
    scope T[] work,
    scope void delegate(
        scope const(T)[] x,
        scope T[] z,
    ) @safe pure nothrow @nogc projection,
    scope QPTerminationCriteria delegate(
        scope const(T)[] x,
        scope const(T)[] xPrev,
        scope const(T)[] y,
        scope const(T)[] yPrev,
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
    auto qCurr = q.sliced;
    auto xCurr = x.sliced;
    auto yCurr = y.sliced;
    auto zCurr = z.sliced;
    auto xPrev = work[n * 0 .. n * 1].sliced;
    auto yPrev = work[n * 1 .. n * 2].sliced;
    xPrev[] = xCurr;
    zCurr[] = xCurr;
    yPrev[] = yCurr;
    int rhoAge;
    T qMax = T(0).reduce!fmax(map!fabs(qCurr));
    T rho = settings.initialRho;
    with(settings) foreach (i; 0 .. maxIterations)
    {
        rho = rho.fmin(maxRho).fmax(minRho);

        xCurr[] = GoldenRatio * (1 / (1 + rho)) * (rho * zCurr - yPrev - qCurr);
        yCurr[] = xCurr + (1 - GoldenRatio) * zCurr + 1 / rho * yPrev;
        xCurr[] += (1 - GoldenRatio) * xPrev;
        projection(y, z);
        yCurr[] = rho * (yCurr - zCurr);

        T xMax = T(0).reduce!fmax(map!fabs(xCurr));
        T yMax = T(0).reduce!fmax(map!fabs(yCurr));
        T zMax = T(0).reduce!fmax(map!fabs(zCurr));
        T primResidual = T(0).reduce!fmax(map!fabs(xCurr - zCurr));
        T dualResidual = T(0).reduce!fmax(map!fabs(xCurr + yCurr + qCurr));
        T primScale = fmax(xMax, zMax);
        T dualScale = fmax(qMax, fmax(xMax, yMax));
        T primResidualNormalized = primResidual ? primResidual / primScale : 0;
        T dualResidualNormalized = dualResidual ? dualResidual / dualScale : 0;

        T epsPrim = epsAbs + epsRel * primScale;
        T epsDual = epsAbs + epsRel * dualScale;
        if (primResidual <= epsPrim && dualResidual <= epsDual)
            return QPTerminationCriteria.solved;

        if (infeasibilityTolerance)
            if (auto criteria = infeasibilityTolerance(x, xPrev.field, y, yPrev.field))
                return criteria;

        if (adaptiveRho && ++rhoAge >= minAdaptiveRhoAge)
        {
            T newRho = rho * sqrt(primResidualNormalized / dualResidualNormalized);
            if (fmax(rho, newRho) > 5 * fmin(rho, newRho))
            {
                zCurr[] = xCurr;
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
            sytrs_3!T('U', F, F_e, F_ipiv, x.sliced(1, n));
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
