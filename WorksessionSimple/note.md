We are integrating
$$
\int_{z_L}^{z_R} (a_j + b_j z)\,\phi(z)\,dz,
$$
which is the expectation of the line $a_j + b_j z$ weighted by the normal density between $z_L$ and $z_R$.

Now split it:
$$
a_j \int_{z_L}^{z_R} \phi(z)\,dz
+ b_j \int_{z_L}^{z_R} z\,\phi(z)\,dz.
$$

We know two Gaussian identities:
$$
\int \phi(z)\,dz = \Phi(z), \qquad
\int z\,\phi(z)\,dz = -\phi(z).
$$

So:
$$
\int_{z_L}^{z_R} \phi(z)\,dz = \Phi(z_R) - \Phi(z_L),
\qquad
\int_{z_L}^{z_R} z\,\phi(z)\,dz = \phi(z_L) - \phi(z_R).
$$

Plug them in:
$$
\int_{z_L}^{z_R} (a_j + b_j z)\,\phi(z)\,dz
= a_j \bigl(\Phi(z_R) - \Phi(z_L)\bigr)
+ b_j \bigl(\phi(z_L) - \phi(z_R)\bigr).
$$
