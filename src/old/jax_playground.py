#%% first derivative (gradient)

import jax
import jax.numpy as jnp

f = lambda x: jnp.sin(x) * jnp.exp(-x)
df = jax.grad(f)
print(df(1.0))


# %% jacobian

def f(x):  # x is a vector
    return jnp.array([x[0]**2 + x[1], jnp.sin(x[1])])

jacobian_f = jax.jacfwd(f)
print(jacobian_f(jnp.array([1.0, 2.0])))


# %% Hessian

def f(x):  # x is a vector
    return jnp.dot(x, x) + jnp.sin(x[0])

hessian_f = jax.hessian(f)
print(hessian_f(jnp.array([1.0, 2.0])))


# %% batching with vmap (vectorized map)

xs = jnp.linspace(-2.0, 2.0, 10)
f = lambda x: x**2 + jnp.sin(x)

# Vectorized application
vf = jax.vmap(f)
print(vf(xs))

#%% Composition: Grad of Grad 

f = lambda x: jnp.sin(x) * jnp.exp(-x)
grad_f = jax.grad(f)
second_derivative = jax.grad(grad_f)

print(second_derivative(1.0))

#%% Transformations (jit, grad, vmap combined)
@jax.jit
@jax.grad
def f(x):
    return jnp.tanh(x**2)

print(f(2.0))

# %%
