from email import message


class Logger:
    tab = '    '
    def __init__(self, save_to_path=None):
        self.save_to_path = save_to_path
        if save_to_path is not None:
            open(save_to_path, 'w').close()


    def print_or_save(self, message):
        print(message)
        if self.save_to_path is not None:
            file = open(self.save_to_path, "a") 
            file.write(f"{message}\n")
            file.close()


    def log_losses(self, losses):
        self.print_or_save(f"{self.tab}{self.tab}Losses")
        messages = [f"{self.tab}{self.tab}{self.tab}{key}: {losses[key]}" for key in losses]
        self.print_or_save('\n'.join(messages))

    
    def log_evaluation_metrics(self, metrics):
        self.print_or_save(f"{self.tab}Metrics")
        messages = [f"{self.tab}{self.tab}{key}: {metrics[key]}" for key in metrics]
        self.print_or_save('\n'.join(messages))


    def log_epoch_start(self, epoch, n_epoch):
        message = f"{self.tab}-------------------- Epoch {epoch} / {n_epoch} --------------------"
        self.print_or_save(message)


    def log_schedule_start(self, schedule):
        message = f"{self.tab}Schedule: {schedule.name}"
        self.print_or_save(message)

    
    def log_method_start(self, method):
        message = f"{method.upper()}"
        self.print_or_save(message)