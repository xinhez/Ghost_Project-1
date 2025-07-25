from tabulate import tabulate


class Logger:
    tab = "    "

    def __init__(self, save_log_path, verbose):
        self.save_log_path = save_log_path
        self.verbose = verbose

    def log_message(self, message):
        if self.verbose:
            print(message)
        if self.save_log_path is not None:
            file = open(self.save_log_path, "a")
            file.write(f"{message}\n")
            file.close()

    def log_losses(self, losses):
        self.log_message(f"{self.tab}Losses")
        messages = [
            f"{self.tab}{self.tab}{key}: {losses[key]}" for key in sorted(losses.keys())
        ]
        self.log_message("\n".join(messages))

    def log_metrics(self, metrics):
        self.log_message(f"{self.tab}Metrics")
        messages = []
        for key in sorted(metrics.keys()):
            if isinstance(metrics[key], list):
                header = f"{self.tab}{self.tab}{key}: "
                metric = "\n".join(
                    [
                        " " * len(header) + line
                        for line in tabulate(metrics[key]).split("\n")
                    ]
                ).strip()
                messages.append(header + metric)
            else:
                messages.append(f"{self.tab}{self.tab}{key}: {metrics[key]}")
        self.log_message("\n".join(messages))

    def log_epoch_start(self, epoch, n_epoch):
        message = f"\n{self.tab}(Epoch {epoch} / {n_epoch})"
        self.log_message(message)

    def log_save_model(self, name):
        message = f"{self.tab}{self.tab}{self.tab}Saving model to {name}"
        self.log_message(message)

    def log_schedule_start(self, schedule):
        message = f"{self.tab}========== Schedule {schedule.order}: {schedule.name} =========="
        self.log_message(message)

    def log_method_start(self, method, task=None):
        message = f"\n{method.upper()} {''if task is None else 'Task:'} {'' if task is None else task}"
        self.log_message(message)
