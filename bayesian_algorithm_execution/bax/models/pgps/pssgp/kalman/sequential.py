import tensorflow as tf

__all__ = ["kf", "ks", "kfs"]

from tensorflow_probability.python.distributions import MultivariateNormalTriL

mv = tf.linalg.matvec
mm = tf.linalg.matmul


def kf(lgssm, observations, return_loglikelihood=False, return_predicted=False):
    P0, Fs, Qs, H, R = lgssm
    dtype = P0.dtype
    m0 = tf.zeros(tf.shape(P0)[0], dtype=dtype)

    def body(carry, inp):
        ell, m, P, *_ = carry
        y, F, Q = inp
        mp = mv(F, m)
        Pp = F @ mm(P, F, transpose_b=True) + Q
        Pp = 0.5 * (Pp + tf.transpose(Pp))

        def update(m, P, ell):
            S = H @ mm(P, H, transpose_b=True) + R
            yp = mv(H, m)
            chol = tf.linalg.cholesky(S)
            predicted_dist = MultivariateNormalTriL(yp, chol)
            ell_t = predicted_dist.log_prob(y)
            Kt = tf.linalg.cholesky_solve(chol, H @ P)

            m = m + mv(Kt, y - yp, transpose_a=True)
            P = P - mm(Kt, S, transpose_a=True) @ Kt
            ell = ell + ell_t
            return ell, m, P

        nan_y = ~tf.math.is_nan(y)
        nan_res = (ell, mp, Pp)
        ell, m, P = tf.cond(nan_y, lambda: update(mp, Pp, ell), lambda: nan_res)
        P = 0.5 * (P + tf.transpose(P))
        return ell, m, P, mp, Pp

    ells, fms, fPs, mps, Pps = tf.scan(body,
                                       (observations, Fs, Qs),
                                       (tf.constant(0., dtype), m0, P0, m0, P0))
    returned_values = (fms, fPs) + ((ells[-1],) if return_loglikelihood else ()) + (
        (mps, Pps) if return_predicted else ())
    return returned_values


def ks(lgssm, ms, Ps, mps, Pps):
    _, Fs, Qs, *_ = lgssm

    def body(carry, inp):
        F, Q, m, P, mp, Pp = inp
        sm, sP = carry

        chol = tf.linalg.cholesky(Pp)
        Ct = tf.linalg.cholesky_solve(chol, F @ P)
        sm = m + mv(Ct, sm - mp, transpose_a=True)
        sP = P + mm(Ct, sP - Pp, transpose_a=True) @ Ct
        return sm, 0.5 * (sP + tf.transpose(sP))

    (sms, sPs) = tf.scan(body,
                         (Fs[1:], Qs[1:], ms[:-1], Ps[:-1], mps[1:], Pps[1:]),
                         (ms[-1], Ps[-1]), reverse=True)
    sms = tf.concat([sms, tf.expand_dims(ms[-1], 0)], 0)
    sPs = tf.concat([sPs, tf.expand_dims(Ps[-1], 0)], 0)
    return sms, sPs


def kfs(model, observations):
    fms, fPs, mps, Pps = kf(model, observations, return_predicted=True)
    return ks(model, fms, fPs, mps, Pps)
