def inside_pgd(self, X, y, width, height, alpha, num_iter, xskip, yskip, out_j, out_i, random=False):
    """Optimize the sticker content at a specific location"""
    model = self.base_classifier
    model.eval()
    
    # Step 1: Create a binary mask for sticker location
    sticker = torch.zeros(X.shape, requires_grad=True)
    for num, ii in enumerate(out_i):
        j = int(out_j[num].item())
        i = int(ii.item())
        # Mark sticker region as 1, rest as 0
        sticker[num, :, yskip*j:(yskip*j+height), xskip*i:(xskip*i+width)] = 1
    sticker = sticker.to(y.device)

    # Step 2: Initialize sticker content
    if random == False:
        delta = torch.zeros_like(X, requires_grad=True) + 1/2  # Gray initialization
    else:
        delta = torch.rand_like(X, requires_grad=True).to(y.device)
        delta.data = delta.data * 255

    # Step 3: Create composite image
    X1 = torch.rand_like(X, requires_grad=True).to(y.device)
    # Original image in non-sticker regions + delta in sticker region
    X1.data = X.detach() * (1 - sticker) + (delta.detach() * sticker)
    
    # Step 4: Optimize sticker pixels using PGD
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X1), y)
        loss.backward()
        
        # Update only sticker pixels (gradient is masked by sticker)
        X1.data = (X1.detach() + alpha * X1.grad.detach().sign() * sticker)
        # Clamp to [0, 1] range
        X1.data = (X1.detach()).clamp(0, 1)
        X1.grad.zero_()
        
    return (X1).detach()