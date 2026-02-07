# Inside training loop for traffic signs
for inputs, labels in dataloaders[phase]:
    # Create ROA module (32 is the image size for traffic signs)
    roa = ROA(model, 32)
    
    with torch.set_grad_enabled(search==1):
        if search == 0:
            # Exhaustive search
            inputs = roa.exhaustive_search(
                inputs, labels, args.alpha, args.iters, 
                args.width, args.height, args.stride, args.stride
            )
        else:
            # Gradient-based search (faster, uses top-k candidates)
            inputs = roa.gradient_based_search(
                inputs, labels, args.alpha, args.iters, 
                args.width, args.height, args.stride, args.stride, 
                args.nums_choose  # Number of potential positions to consider
            )

        save_image('1112sticker_'+str(args.width)+str(args.iters), inputs)
        
    optimizer.zero_grad()
    
    with torch.set_grad_enabled(phase == 'train'):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels = labels.to(device)
        loss = criterion(outputs, labels)
        
        if phase == 'train':
            loss.backward()
            optimizer.step()
            
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)