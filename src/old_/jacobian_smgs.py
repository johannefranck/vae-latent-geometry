import torch
# ------------------ Jacobian & Energy ------------------

def compute_jacobian(decoder, z):
    z = z.detach().clone().requires_grad_(True)
    x = decoder(z).mean.view(-1)
    grads = [torch.autograd.grad(x[i], z, retain_graph=True, create_graph=True)[0] for i in range(x.shape[0])]
    return torch.cat(grads, dim=0)  # (output_dim, dim)


def compute_energy(spline, decoder, t_vals):
    z = spline(t_vals)
    dz = (z[1:] - z[:-1]) * t_vals.shape[0]
    dz = dz.unsqueeze(1)  # (T-1, 1, dim)

    G_all = []
    for zi in z[:-1]:
        J = compute_jacobian(decoder, zi.unsqueeze(0))  # (output_dim, dim)
        identity = torch.eye(J.shape[1], device=J.device)
        deviation = torch.norm(J.T @ J - identity)
        #print("Deviation from Euclidean metric:", deviation.item())
        if J.norm().item() < 1:
            print("Jacobian norm:", J.norm().item())

        G = J.T @ J  # (dim, dim)
        G_all.append(G.unsqueeze(0))
    G_all = torch.cat(G_all, dim=0)  # (T-1, dim, dim)

    energy = torch.bmm(torch.bmm(dz, G_all), dz.transpose(1, 2))  # (T-1, 1, 1)
    return energy.mean()