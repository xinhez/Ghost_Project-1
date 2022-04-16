class Logger:
    tab = '    '
    def __init__(self, save_log_path=None):
        self.save_log_path = save_log_path


    def print_or_save(self, message):
        print(message)
        if self.save_log_path is not None:
            file = open(self.save_log_path, "a") 
            file.write(f"{message}\n")
            file.close()


    def log_losses(self, losses):
        self.print_or_save(f"{self.tab}Losses")
        messages = [f"{self.tab}{self.tab}{key}: {losses[key]}" for key in sorted(losses.keys())]
        self.print_or_save('\n'.join(messages))

    
    def log_metrics(self, metrics):
        self.print_or_save(f"{self.tab}Metrics")
        messages = [f"{self.tab}{self.tab}{key}: {metrics[key]}" for key in sorted(metrics.keys())]
        self.print_or_save('\n'.join(messages))


    def log_epoch_start(self, epoch, n_epoch):
        message = f"\n{self.tab}(Epoch {epoch} / {n_epoch})"
        self.print_or_save(message)

    
    def log_save_model(self, name):
        message = f"{self.tab}{self.tab}{self.tab}Saving model to {name}"
        self.print_or_save(message)


    def log_schedule_start(self, schedule):
        message = f"{self.tab}========== Schedule {schedule.order}: {schedule.name} =========="
        self.print_or_save(message)

    
    def log_method_start(self, method, task=None):
        message = f"{method.upper()} {''if task is None else 'Task:'} {'' if task is None else task}"
        self.print_or_save(message)