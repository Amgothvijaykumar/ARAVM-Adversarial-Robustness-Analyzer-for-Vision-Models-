if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("model", type=str, help="original(clean) model you want to do DOA")
    parser.add_argument("-alpha", type=int, help="alpha learning rate")
    parser.add_argument("-iters", type=int, help="iterations of PGD")
    parser.add_argument("-out", type=int, help="name of final model")
    parser.add_argument("-search", type=int, 
                       help="method of searching, '0' is exhaustive_search, '1' is gradient_based_search")
    parser.add_argument("-epochs", type=int, help="epochs")
    parser.add_argument("--stride", type=int, default=10, 
                       help="the skip pixels when searching")
    parser.add_argument("--width", type=int, default=70, 
                       help="width of the rectangular occlusion")
    parser.add_argument("--height", type=int, default=70, 
                       help="height of the rectangular occlusion")
    args = parser.parse_args()
    
    print(args)
    
    torch.manual_seed(123456)
    torch.cuda.empty_cache()
    print('output model will locate on ../donemodel/new_sticker_model0'+str(args.out)+'.pt')
    
    # Load data
    dataloaders, dataset_sizes = data_process(batch_size=32)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained clean model
    model_ft = VGG_16() 
    model_ft.load_state_dict(torch.load('../donemodel/'+args.model))
    model_ft.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
    
    # Learning rate scheduler: decay by 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    # Fine-tune with DOA
    model_ft = sticker_train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler,
        args.alpha, args.iters, args.search, num_epochs=args.epochs
    )
    
    # Test and save
    test(model_ft, dataloaders, dataset_sizes)
    torch.save(model_ft.state_dict(), 
              '../donemodel/new_sticker_model0'+str(args.out)+'.pt')