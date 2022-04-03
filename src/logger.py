class Logger():
    def __init__(self, model):
        self.model = model 

    def log_epoch_start(self, epoch, schedule):
        print(f"===== Epoch {epoch} Schedule {schedule.name} =====")