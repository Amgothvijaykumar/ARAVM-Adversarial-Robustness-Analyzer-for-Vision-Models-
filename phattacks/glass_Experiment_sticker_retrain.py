def sticker_train_model(model, criterion, optimizer, scheduler, alpha, iters, search, num_epochs=10):
    """
    DOA Training Function
    Fine-tunes a pre-trained model to be robust against rectangular occlusion attacks
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs[:,[2,1,0],:,:]  # RGB to BGR conversion
                
                # CRITICAL: Create ROA module for generating adversarial training examples
                ROA_module = ROA(model, alpha, iters)
                
                # Step 1: Generate adversarial examples with rectangular occlusions
                with torch.set_grad_enabled(search==1):
                    if search == 0:
                        # Exhaustive search: tries all positions
                        ROA_inputs = ROA_module.exhaustive_search(
                            inputs, labels, args.width, args.height, 
                            args.stride, args.stride
                        )
                    else:
                        # Gradient-based search: uses gradients to find worst positions
                        ROA_inputs = ROA_module.gradient_based_search(
                            inputs, labels, args.width, args.height, 
                            args.stride, args.stride
                        )

                optimizer.zero_grad()
                
                if phase == 'train':
                    model.train()
                    
                # Step 2: Train on adversarial examples
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass with occluded images
                    outputs = model(ROA_inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = labels.to(device)
                    
                    # Compute loss - model learns to classify correctly despite occlusion
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model_ft.state_dict(), 
                          '../donemodel/new_sticker_model0'+str(args.out)+'.pt')
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model