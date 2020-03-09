/++
+/
module mir.optim.boxcqp;

import mir.ndslice;
import lapack: lapackint;
import mir.optim.internal;

import mir.math.sum;
import mir.utility: min, max;
import mir.math.common: fmin, fmax, fabs, sqrt;
import mir.internal.memory;
import mir.blas;
import mir.lapack;
import cblas;

@safe pure nothrow @nogc
bool solveQP(T)(
    Slice!(const(T)*, 2, Canonical) P,
    Slice!(const(T)*) q,
    Slice!(const(T)*) l,
    Slice!(const(T)*) u,
    Slice!(T*) x,
    T tolActiveSet = T.epsilon.sqrt,
    size_t maxIteratios = 0,
    )
    if (is(T == float) || is(T == double))
{
    enum Flag : byte
    {
        l,
        u,
        s,
    }

    auto n = q.length;

    if (!maxIteratios)
        maxIteratios = n * 10 + 100; // fix

    auto rcwork = rcslice!T(n ^^ 2 * 2 + n * 8);
    auto flags = rcslice!Flag(n);
    auto iWork = rcslice!lapackint(n);
    auto bufferG = rcwork.lightScope;

    auto lambda  = bufferG[0 .. n]; bufferG = bufferG[n .. $];
    auto mu  = bufferG[0 .. n]; bufferG = bufferG[n .. $];

    lambda[] = 0;
    mu[] = 0;

    {
        auto buffer = bufferG;
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
            iWork.lightScope);

        if (info != 0 && info != n + 1)
        {
            // throw exception
        }
    }
    
    bool convergence = true;

    foreach (i; 0 .. n)
    {
        convergence = convergence && (l[i] <= x[i] && x[i] <= u[i] );
    }
    
    for (size_t step; !convergence && step < maxIteratios; ++step)
    {
        size_t s;

        foreach (i; 0 .. n)
        {
            if (x[i] < l[i] || x[i] - l[i] < tolActiveSet && lambda[i] >= 0)
            {
                flags[i] = Flag.l;
                x[i] = l[i];
                mu[i] = 0;
            }
            else if ( x[i] > u[i] || u[i] - x[i] < tolActiveSet && mu[i] >= 0)
            {
                flags[i] = Flag.u;
                x[i] = u[i];
                lambda[i] = 0;
            }
            else
            {
                flags[i] = Flag.s;
                iWork[s++] = cast(lapackint)i;
                mu[i]  = 0;
                lambda[i]  = 0;
            }
        }

        auto SIWorkspace = iWork.lightScope[0 .. s];
        auto buffer = bufferG;
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
        {
            // throw exception
        }

        {
            size_t ii;
            foreach (i; 0 .. n) if (flags[i] == Flag.s)
                x[i] = sX[ii];
        }

        foreach (i; 0 .. n) if (flags[i] == Flag.l)
            lambda[i] = dot!T(P[i], x) + q[i];

        foreach (i; 0 .. n) if (flags[i] == Flag.u)
            mu[i] = -(dot!T(P[i], x) + q[i]);

        convergence = true;
        foreach (i; 0 .. n)
        {
            final switch (flags[i])
            {
                case Flag.l: if (lambda[i] >= 0) continue; else break;
                case Flag.u: if (mu[i] >= 0) continue; else break;
                case Flag.s: if (x[i] >= l[i] && x[i] <= u[i]) continue; else break;
            }
            convergence = false;
            break;
        } 
    }

    if (convergence)
        foreach (i; 0 .. n)
            x[i] = x[i].fmin(u[i]).fmax(l[i]);
    
    return convergence;
}

///
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
