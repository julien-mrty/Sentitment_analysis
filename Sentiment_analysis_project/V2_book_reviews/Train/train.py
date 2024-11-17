import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import time


def train_validate_model(model, train_data_loader, validation_data_loader, num_epochs, criterion, optimizer, scheduler,
                         training_logger, print_freq):  # Added weight_decay parameter

    print("Beginning of training...\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_start_time = time.time()  # Start time for the entire training process

    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience = 2  # Number of epochs to wait before stopping
    trigger_times = 0

    for epoch in range(num_epochs):

        start_time = time.time()

        """ TRAIN MODEL """
        train_epoch_loss, train_model_output, train_target_values, current_lr = train_one_epoch(model, train_data_loader, criterion,
                                                                                    optimizer, scheduler, device,
                                                                                    print_freq)
        train_avg_loss = train_epoch_loss / len(train_data_loader)
        print(f"TRAIN : Epoch {epoch + 1}, Loss: {train_avg_loss:.4f}")

        # Convert vectors to class labels
        train_target_labels = np.argmax(train_target_values, axis=1)
        train_model_labels = np.argmax(train_model_output, axis=1)
        num_labels = max(train_target_labels)

        train_report = classification_report(train_target_labels, train_model_labels, labels=range(0, num_labels + 1),
                                             zero_division=0, output_dict=True)
        train_confusion_matrix = confusion_matrix(train_target_labels, train_model_labels)

        """ TEST MODEL """
        val_loss, val_model_predictions, val_target_values = validate_model(model, validation_data_loader, criterion,
                                                                            device, print_freq)
        val_avg_loss = val_loss / len(validation_data_loader)
        print(f"VALIDATION : Epoch {epoch + 1}, Loss : {val_avg_loss:.4f}")

        # Convert vectors to class labels
        val_target_labels = np.argmax(val_target_values, axis=1)
        val_model_labels = np.argmax(val_model_predictions, axis=1)
        num_labels = max(val_target_labels)

        validation_report = classification_report(val_target_labels, val_model_labels, labels=range(0, num_labels + 1),
                                                  zero_division=0, output_dict=True)
        val_confusion_matrix = confusion_matrix(val_target_labels, val_model_labels)

        # Log the epoch information, including imbalance metrics
        training_logger.log_epoch(epoch, current_lr, train_avg_loss, train_report, val_avg_loss, validation_report,
                                  train_confusion_matrix, val_confusion_matrix)

        end_time = time.time()  # Record the end time of the epoch
        epoch_duration = end_time - start_time  # Calculate the duration of the epoch
        total_samples = len(train_data_loader.dataset) + len(validation_data_loader.dataset)  # Total samples processed
        avg_time_per_sample = epoch_duration / total_samples  # Calculate average time per sample

        print(f"======== Epoch {epoch + 1} epoch duration: {epoch_duration:.2f}sec, per sample : {avg_time_per_sample:.6f}sec")

        # Early stopping check
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model_state_dict.pth')
        else:
            trigger_times += 1
            print(f'EarlyStopping: {trigger_times} out of {patience}')
            if trigger_times >= patience:
                print('Early stopping!')
                break


    total_end_time = time.time()  # End time for the entire training process
    total_training_time = total_end_time - total_start_time  # Calculate total training time

    print(f"\nTotal training time: {total_training_time / 60:.2f} minutes.")  # Print total training time in minutes

    return model, training_logger


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, print_freq):
    model.train()

    epoch_loss = 0
    model_output_list = []
    target_values_list = []

    total_samples = len(dataloader.dataset)  # Total number of samples in the epoch
    sample_count = 0  # Counter to track the number of samples processed

    for batch_idx, (input_ids, attention_mask, review_helpfulness, score) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        review_helpfulness = review_helpfulness.to(device)
        score = score.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, review_helpfulness)
        loss = criterion(outputs, score)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        epoch_loss += loss.item()

        # Detach and convert to numpy for sklearn compatibility
        model_output_list.extend(outputs.detach().cpu().numpy())
        target_values_list.extend(score.detach().cpu().numpy())

        # Update sample count
        sample_count += len(input_ids)

        # Print the loss and progress at the specified print frequency
        if (batch_idx + 1) % print_freq == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            current_lr = scheduler.get_last_lr()[0]
            print(f"TRAIN : Batch {batch_idx + 1}/{len(dataloader)} : Loss = {avg_loss:.4f}, "
                  f"LR = {current_lr:.1e}, Processed Samples: {sample_count}/{total_samples}")

    return epoch_loss, model_output_list, target_values_list, current_lr


def validate_model(model, dataloader, criterion, device, print_freq):
    model.eval()

    val_loss = 0
    model_predictions = []
    target_values = []
    total_samples = len(dataloader.dataset)  # Total number of samples in the validation set
    sample_count = 0  # Counter to track the number of samples processed

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, review_helpfulness, score) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            review_helpfulness = review_helpfulness.to(device)
            score = score.to(device)

            outputs = model(input_ids, attention_mask, review_helpfulness)
            loss = criterion(outputs, score)

            val_loss += loss.item()

            # Detach and convert to numpy for sklearn compatibility
            model_predictions.extend(outputs.detach().cpu().numpy())
            target_values.extend(score.detach().cpu().numpy())

            # Update sample count
            sample_count += len(input_ids)

            # Print progress at the end of each batch
            if (batch_idx + 1) % print_freq == 0:
                avg_loss = val_loss / (batch_idx + 1)
                print(f"VALIDATION : Batch {batch_idx + 1}/{len(dataloader)} : Loss = {avg_loss:.4f}, "
                      f"Processed Samples: {sample_count}/{total_samples}")

    return val_loss, model_predictions, target_values