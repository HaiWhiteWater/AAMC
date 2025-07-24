from mmcv.runner.hooks import HOOKS, Hook
import torch


# Define a custom Hook class that inherits from mmcv.runner.hooks.Hook
@HOOKS.register_module()
class UpdateAlphaHook_down(Hook):
    def __init__(self, update_interval=1, decay_rate=0.727):
        # self.model = model
        self.update_interval = update_interval  # alpha interval
        self.decay_rate = decay_rate  # alpha decay rate
        # self.factor = factor

    def before_train_epoch(self, runner):
        model = runner.model  # Get the model from the runner
        # Check if roi_head.bbox_head is ModuleList, this allows Hook to be used with Cascade R-CNN
        if isinstance(model.module.roi_head.bbox_head, torch.nn.ModuleList):
            # Iterate through all bbox_heads
            for bbox_head in model.module.roi_head.bbox_head:
                self.update_alpha_for_loss(bbox_head.loss_cls, runner)
        else:
            # For non-ModuleList cases (like Mask R-CNN), update directly
            loss_func = model.module.roi_head.bbox_head.loss_cls  # Get the classification loss function object from the model
            self.update_alpha_for_loss(loss_func, runner)

        # loss_func = model.module.roi_head.bbox_head.loss_cls  # Get the classification loss function object from the model
        # current_alpha = loss_func.alpha  # Get the current value of alpha parameter in the loss function that needs to be updated
        # start_alpha = loss_func.start_alpha  # Get the initial alpha value from the loss function
        # runner.logger.info(f"before_train_epoch: start_alpha：{start_alpha}, current_alpha: {current_alpha} at epoch {runner.epoch + 1}")
        # if runner.epoch % self.update_interval == 0:
        #     new_alpha = self.alpha_update_function(start_alpha, runner.epoch, self.decay_rate)
        #     loss_func.alpha = new_alpha
        #     runner.logger.info(f"Updated Alpha parameter to {new_alpha} at epoch {runner.epoch + 1}")
        #     runner.logger.info(f"before_train_epoch: now real Alpha : {loss_func.alpha} ")

    # def after_train_epoch(self, runner):
    #     model = runner.model
    #     loss_func = model.module.roi_head.bbox_head.loss_cls
    #     runner.logger.info(f"after_train_epoch: Alpha : {loss_func.alpha} ")
    # if runner.epoch % self.update_interval == 0:
    #     # current_alpha = model.module.roi_head.bbox_head.loss_cls.alpha
    #     start_alpha = loss_func.start_alpha
    #     new_alpha = self.alpha_update_function(start_alpha, runner.epoch, self.decay_rate)
    #     # model.roi_head.bbox_head.loss_cls.alpha = new_alpha
    #     loss_func.alpha = new_alpha
    #     runner.logger.info(f"Updated Alpha parameter to {new_alpha} at epoch {runner.epoch + 2}")
    #     runner.logger.info(f"after_train_epoch: now real Alpha : {loss_func.alpha} ")

    def update_alpha_for_loss(self, loss_func, runner):
        current_alpha = loss_func.alpha  # Get current alpha value
        start_alpha = loss_func.start_alpha  # Get initial alpha value
        runner.logger.info(f"start_alpha: {start_alpha}, current_alpha: {current_alpha} at epoch {runner.epoch}")

        if runner.epoch % self.update_interval == 0:
            # Update alpha value
            # new_alpha = self.alpha_update_function(start_alpha, runner.epoch, self.decay_rate)
            new_alpha = self.alpha_update_function(current_epoch=runner.epoch, start_alpha=start_alpha,
                                                   decay_rate=self.decay_rate)
            loss_func.alpha = new_alpha  # Set new alpha value
            runner.logger.info(f"Updated Alpha parameter to {new_alpha} at epoch {runner.epoch + 1}")
            runner.logger.info(f"now the real Alpha is: {loss_func.alpha}")

    @staticmethod
    def alpha_update_function(current_epoch, start_alpha, decay_rate):
        # decay_value = (decay_rate**current_epoch) * start_alpha
        decay_value = decay_rate * current_epoch
        new_alpha = start_alpha - decay_value
        if (new_alpha < 0):
            new_alpha = 0

        return new_alpha

    # @staticmethod
    # def alpha_update_function(start_alpha, current_epoch, decay_rate):
    #     # decay_value = (decay_rate**current_epoch) * start_alpha
    #     decay_value = decay_rate * current_epoch
    #     new_alpha = start_alpha - decay_value
    #     if (new_alpha < 0):
    #         new_alpha = 0
    #
    #     return new_alpha

    # @staticmethod
    # def alpha_update_function(start_alpha, current_epoch, total_epochs):
    #     # Calculate the decay_rate to reduce per epoch
    #     decay_rate_per_epoch = (start_alpha - 0) / total_epochs
    #     # Calculate new alpha value based on current epoch
    #     new_alpha = start_alpha - (decay_rate_per_epoch * current_epoch)
    #     # Ensure alpha is not less than 0
    #     if new_alpha < 0:
    #         new_alpha = 0
    #     return new_alpha
