module mir.optim.internal;

import mir.ndslice;

static import lapack;
public import lapack: lapackint;
import mir.blas;
import mir.utility: min, max;

alias T = double;

enum mallocErrorMsg = "Failed to allocate memory.";
version(D_Exceptions)
    static immutable mallocError = new Error(mallocErrorMsg);

auto safeAlloc(T)(size_t n) @trusted
{
    import mir.internal.memory;
    auto workPtr = cast(T*)malloc(T.sizeof * n);
    if (workPtr is null)
    {
        version(D_Exceptions)
            throw mallocError;
        else
            assert(0, msg);
    }
    return workPtr[0 .. n];
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

@safe pure nothrow @nogc
void eigenTimes(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValues,
    Slice!(const(T)*) x,
    Slice!(T*) temp,
    Slice!(T*) y,
    T alpha = 1,
)
{
    svdTimes(eigenVectors, eigenVectors, eigenValues, x, temp, y, alpha);
}

@safe pure nothrow @nogc
void eigenSolve(
    Slice!(const(T)*, 2, Canonical) eigenVectors,
    Slice!(const(T)*) eigenValues,
    Slice!(const(T)*) y,
    Slice!(T*) temp,
    Slice!(T*) x,
    T alpha = 1,
)
{
    svdSolve(eigenVectors, eigenVectors, eigenValues, y, temp, x, alpha);
}

@safe pure nothrow @nogc
void eigenSqrtTimes(
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
do {
    auto r = eigenValuesRoots.length;
    gemv!T(1, eigenVectors[0 .. r], x, 0, y);
    y.mulBy(eigenValuesRoots);
}

@safe pure nothrow @nogc
void eigenSqrtSolve(
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
do {
    auto r = eigenValuesRoots.length;
    y.divBy(eigenValuesRoots);
    gemv!T(1, eigenVectors[0 .. r].transposed, y, 0, x);
}

@safe pure nothrow @nogc
void eigenSqrtSplit(
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
do {
    auto r = eigenValuesRoots.length;
    gemv!T(1, eigenVectors[0 .. r], x, 0, y);
    y.divBy(eigenValuesRoots);
}

@safe pure nothrow @nogc
void eigenSqrtSplitReverse(
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
do {
    auto r = eigenValuesRoots.length;
    y.mulBy(eigenValuesRoots);
    gemv!T(1, eigenVectors.transposed[0 .. r], y, 0, x);
}

@safe pure nothrow @nogc
void svdTimes(
    Slice!(const(T)*, 2, Canonical) leftSingularVectors,
    Slice!(const(T)*, 2, Canonical) rightSingularVectors,
    Slice!(const(T)*) singularValues,
    Slice!(const(T)*) x,
    Slice!(T*) temp,
    Slice!(T*) y,
    T alpha = 1,
)
in {
    assert(leftSingularVectors.length!0 == leftSingularVectors.length!1);
    assert(rightSingularVectors.length!0 == rightSingularVectors.length!1);
    assert(singularValues.length <= min(leftSingularVectors.length, rightSingularVectors.length));
    assert(singularValues.length <= temp.length);
    assert(x.length == rightSingularVectors.length);
    assert(y.length == leftSingularVectors.length);
}
do {
    auto r = singularValues.length;
    temp = temp[0 ..r];
    gemv!T(1, rightSingularVectors[0 .. r].transposed, x, 0, temp);
    temp.mulBy(singularValues);
    gemv!T(alpha, leftSingularVectors[0 .. r], temp, 0, y);
}

@safe pure nothrow @nogc
void svdSolve(
    Slice!(const(T)*, 2, Canonical) leftSingularVectors,
    Slice!(const(T)*, 2, Canonical) rightSingularVectors,
    Slice!(const(T)*) singularValues,
    Slice!(const(T)*) y,
    Slice!(T*) temp,
    Slice!(T*) x,
    T alpha = 1,
)
in {
    assert(leftSingularVectors.length!0 == leftSingularVectors.length!1);
    assert(rightSingularVectors.length!0 == rightSingularVectors.length!1);
    assert(singularValues.length <= min(leftSingularVectors.length, rightSingularVectors.length));
    assert(singularValues.length <= temp.length);
    assert(x.length == rightSingularVectors.length);
    assert(y.length == leftSingularVectors.length);
}
do {
    while(singularValues.length && singularValues[$-1] < T.min_normal)
        singularValues = singularValues[0 .. $ - 1];
    auto r = singularValues.length;
    temp = temp[0 ..r];
    gemv!T(1, leftSingularVectors[0 .. $, 0 .. r], y, 0, temp);
    temp.divBy(singularValues);
    gemv!T(alpha, rightSingularVectors[0 .. $, 0 .. r].transposed, temp, 0, x);
}

version(LAPACK_STD_COMPLEX)
{
    import std.complex: Complex;
    alias _cfloat = Complex!float;
    alias _cdouble = Complex!double;
}
else
{
    alias _cfloat = cfloat;
    alias _cdouble = cdouble;
}

void copyMinor(size_t N, IteratorFrom, SliceKind KindFrom, IteratorTo, SliceKind KindTo, IndexIterator)(
    Slice!(IteratorFrom, N, KindFrom) from,
    Slice!(IteratorTo, N, KindTo) to,
    Slice!IndexIterator[N] indexes...
)
in {
    import mir.internal.utility: Iota;
    static foreach (i; Iota!N)
        assert(indexes[i].length == to.length!i);
}
do {
    static if (N == 1)
        to[] = from[indexes[0]];
    else
    foreach (i; 0 .. indexes[0].length)
    {
        copyMinor!(N - 1)(from[indexes[0][i]], to[i], indexes[1 .. N]);
    }
}

///
@safe pure nothrow
unittest
{
    import mir.ndslice;
    //  0  1  2  3
    //  4  5  6  7
    //  8  9 10 11
    auto a = iota!int(3, 4);
    auto b = slice!int(2, 2);
    copyMinor(a, b, [2, 1].sliced, [0, 3].sliced);
    assert(b == [[8, 11], [4, 7]]);
}

void reverseInPlace(Iterator)(Slice!Iterator slice)
{
    import mir.utility : swap;
    foreach (i; 0 .. slice.length / 2)
        swap(slice[i], slice[$ - (i + 1)]);
}

unittest
{
    import mir.ndslice;
    auto s = 5.iota.slice;
    s.reverseInPlace;
    assert([4, 3, 2, 1, 0]);
}

extern(C) @system @nogc nothrow
{
/// Computes the factorization of a real symmetric-indefinite matrix,
/// using the diagonal pivoting method.
void ssytrf_rk_(ref const char uplo, ref const lapackint n, float *a, ref const lapackint lda, float* e, lapackint *ipiv, float *work, ref lapackint lwork, ref lapackint info);
void dsytrf_rk_(ref const char uplo, ref const lapackint n, double *a, ref const lapackint lda, double* e, lapackint *ipiv, double *work, ref lapackint lwork, ref lapackint info);
void csytrf_rk_(ref const char uplo, ref const lapackint n, _cfloat *a, ref const lapackint lda, _cfloat* e, lapackint *ipiv, _cfloat *work, ref lapackint lwork, ref lapackint info);
void zsytrf_rk_(ref const char uplo, ref const lapackint n, _cdouble *a, ref const lapackint lda, _cdouble* e, lapackint *ipiv, _cdouble *work, ref lapackint lwork, ref lapackint info);

/// Solves a real symmetric indefinite system of linear equations AX=B,
/// using the factorization computed by SSPTRF_RK.
void ssytrs_3_(ref const char uplo, ref const lapackint n, ref const lapackint nrhs, const(float) *a, ref const lapackint lda, const(float)* e, const(lapackint)* ipiv, float* b, ref const lapackint ldb, ref lapackint info);
void dsytrs_3_(ref const char uplo, ref const lapackint n, ref const lapackint nrhs, const(double) *a, ref const lapackint lda, const(double)* e, const(lapackint)* ipiv, double* b, ref const lapackint ldb, ref lapackint info);
void csytrs_3_(ref const char uplo, ref const lapackint n, ref const lapackint nrhs, const(_cfloat) *a, ref const lapackint lda, const(_cfloat)* e, const(lapackint)* ipiv, _cfloat* b, ref const lapackint ldb, ref lapackint info);
void zsytrs_3_(ref const char uplo, ref const lapackint n, ref const lapackint nrhs, const(_cdouble) *a, ref const lapackint lda, const(_cdouble)* e, const(lapackint)* ipiv, _cdouble* b, ref const lapackint ldb, ref lapackint info);
}

alias sytrf_rk_ = ssytrf_rk_;
alias sytrf_rk_ = dsytrf_rk_;
alias sytrf_rk_ = csytrf_rk_;
alias sytrf_rk_ = zsytrf_rk_;

alias sytrs_3_ = ssytrs_3_;
alias sytrs_3_ = dsytrs_3_;
alias sytrs_3_ = csytrs_3_;
alias sytrs_3_ = zsytrs_3_;

///
size_t sytrs_3(T)(
    char uplo,
    Slice!(const(T)*, 2, Canonical) a,
    Slice!(const(T)*) e,
    Slice!(const(lapackint)*) ipiv,
    Slice!(T*, 2, Canonical) b,
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
// ref char uplo, ref lapackint n, ref lapackint nrhs, float *a, ref lapackint lda, float* e, lapackint *ipiv, float *b, ref lapackint ldb, ref lapackint info
    sytrs_3_(uplo, n, nrhs, a.iterator, lda, e.iterator, ipiv.iterator, b.iterator, ldb, info);
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
    sytrf_rk_(uplo, n, a.iterator, lda, e.iterator, ipiv.iterator, work.iterator, lwork, info);
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

version(none):

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
