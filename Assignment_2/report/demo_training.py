from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def fit_demo(X_train, y_train, X_valid, candidate, seed=5303):
    study.setup(seed)
    sx = StandardScaler().fit(X_train)
    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    xt = torch.tensor(sx.transform(X_train), dtype=torch.float32)
    xv = torch.tensor(sx.transform(X_valid), dtype=torch.float32)
    yt = torch.tensor(sy.transform(y_train.reshape(-1, 1)), dtype=torch.float32)
    model = build_model(candidate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate['learning_rate'],
        weight_decay=candidate['weight_decay'])
    autoencoder = candidate['family'] in ('ae_ridge', 'dae_ridge')
    noise = candidate.get('noise_std_standardized', 0)
    for epoch in range(candidate['epoch']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs = xt + noise * torch.randn_like(xt) if noise else xt
        target = xt if autoencoder else yt
        loss = nn.functional.mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        if autoencoder:
            z_train = model.encoder(xt).numpy()
            z_valid = model.encoder(xv).numpy()
        else:
            return sy.inverse_transform(model(xv).numpy()).ravel()
    sz = StandardScaler().fit(z_train)
    regressor = Ridge(alpha=candidate['alpha'])
    regressor.fit(sz.transform(z_train), y_train)
    return regressor.predict(sz.transform(z_valid))
