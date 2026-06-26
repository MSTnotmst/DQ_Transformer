import time
import os
import datetime
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


def _hms(seconds):
    """秒數 → H:MM:SS 字串。"""
    return str(datetime.timedelta(seconds=int(seconds)))

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    # [計時] 累計訓練時間 + ETA，寫到 checkpoints/<name>/train_time.log（append，跨續訓累積）。
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    time_log_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(time_log_dir, exist_ok=True)
    time_log_path = os.path.join(time_log_dir, 'train_time.log')

    def log_time(msg):
        print(msg)
        with open(time_log_path, 'a') as f:
            f.write(msg + '\n')

    train_start_time = time.time()
    log_time('=== 訓練開始 %s | 從 epoch %d 跑到 %d ===' % (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), opt.epoch_count, total_epochs))
    if opt.epoch_count > total_epochs:
        log_time('⚠️ 警告：--epoch_count(%d) > 總 epoch n_epochs+n_epochs_decay(%d)，'
                 '訓練迴圈是空的、不會跑任何 epoch。續訓請把 n_epochs(+decay) 設得比 epoch_count 大。'
                 % (opt.epoch_count, total_epochs))

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset visualizer: make sure it saves results to HTML at least once every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            else:
                t_data = 0

            total_iters += 1  # opt.batch_size
            epoch_iter += 1  # opt.batch_size

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if (total_iters % opt.print_freq == 0) or (
                    total_iters == 1):  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # [計時] 累計時間 / 平均每 epoch / 預估剩餘（ETA 以本次執行的平均速度估算）。
        elapsed = time.time() - train_start_time
        done = epoch - opt.epoch_count + 1
        avg = elapsed / max(done, 1)
        eta = avg * (total_epochs - epoch)
        log_time('End of epoch %d/%d | 本 epoch %ds | 累計 %s | 平均 %.1fs/epoch | ETA 約 %s' % (
            epoch, total_epochs, time.time() - epoch_start_time, _hms(elapsed), avg, _hms(eta)))
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.

    ran_epochs = total_epochs - opt.epoch_count + 1
    if ran_epochs <= 0:
        log_time('=== 結束（沒有訓練任何 epoch，見上面警告） %s ===' %
                 datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    else:
        log_time('=== 訓練結束 %s | 本次跑了 %d epoch | 總耗時 %s ===' % (
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ran_epochs, _hms(time.time() - train_start_time)))
