def choose_color(model,X,y,glass,mean):
    """Choose the color for the eyeglass frame"""
    model.eval()
    potential_starting_color0 = [128,220,160,200,220]
    potential_starting_color1 = [128,130,105,175,210]
    potential_starting_color2 = [128,  0, 55, 30, 50]

    delta1 = torch.zeros(X.size()).to(y.device)
    
    # Apply color to the glass mask (BGR channels)
    delta1[:,0,:,:] = glass[0,:,:]*potential_starting_color2[0]
    delta1[:,1,:,:] = glass[1,:,:]*potential_starting_color1[0]
    delta1[:,2,:,:] = glass[2,:,:]*potential_starting_color0[0]

    return delta1


def glass_attack(model, X, y, glass, alpha=1, num_iter=20, momentum=0.4):
    """ Construct glass frame adversarial examples on the examples X"""
    model.eval()
    mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
    de = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = mean.to(de)
    
    # Step 1: Create transformed image
    X1 = torch.zeros_like(X, requires_grad=True)
    # Keep original image in non-glass regions, use (1-glass) as mask
    X1.data = (X + mean) * (1 - glass)
    
    # Step 2: Add colored glass frame
    color_glass = choose_color(model, X1, y, glass, mean)
    
    with torch.set_grad_enabled(True):
        # Combine: original image (masked) + colored glass
        X1.data = X1.data + color_glass - mean
        
        delta = torch.zeros_like(X)
    
        # Step 3: Optimize glass pixels using PGD
        for t in range(num_iter):
            # Compute loss
            loss = nn.CrossEntropyLoss()(model(X1), y)
            loss.backward()

            # Extract gradient only in glass region
            delta_change = X1.grad.detach() * glass
            max_val, indice = torch.max(torch.abs(delta_change.view(delta_change.shape[0], -1)), 1)
            r = alpha * delta_change / max_val[:, None, None, None]

            # Apply momentum for smooth updates
            if t == 0:
                delta.data = r
            else:
                delta.data = momentum * delta.detach() + r

            # Update adversarial image
            X1.data = (X1.detach() + delta.detach())
            # Clamp to valid pixel range
            X1.data = (X1.detach() + mean).clamp(0, 255) - mean
            # Round to discrete pixel values
            X1.data = torch.round(X1.detach() + mean) - mean
          
            X1.grad.zero_()

        return (X1).detach()