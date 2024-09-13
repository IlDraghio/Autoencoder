import torch

vali = True

A = AE()

AutoEncoder = A.to(device)

criterion = nn.SmoothL1Loss()

# Definizione optimizer

opt = torch.optim.Adam(params=AutoEncoder.parameters(), lr=1.e-3, weight_decay=1e-5)
epochs = 200
train_loss_list = []
vali_loss_list = []

for epoch in range(epochs):
    t0 = time.time()

    AutoEncoder.train()
    train_loss = 0

    # Training step

    for x in train_dl:

        x=x[0].to(device)

        opt.zero_grad()

        x_recon,_= AutoEncoder(x)
        loss = criterion(x_recon, x)
        train_loss += loss.item()

        loss.backward()
        opt.step()


    train_loss /= len(train_dl)

    train_loss_list.append(train_loss)

    # Validation step

    vali_loss = 0
    if vali:

      AutoEncoder.eval()
      vali_loss = 0

      with torch.no_grad():
        for x in vali_dl:
          x = x[0].to(device)
          x_recon, _ = AutoEncoder(x)
          loss = criterion(x_recon, x)
          vali_loss += loss.item()

      vali_loss /= len(vali_dl)

      vali_loss_list.append(vali_loss)

    elapsed_time = time.time()-t0

    print("epoch: %d, time(s): %.2f, train loss: %.6f, vali loss: %.6f" % (epoch+1, elapsed_time, train_loss, vali_loss))