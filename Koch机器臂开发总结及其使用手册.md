# Koch机器臂开发总结及其使用手册

## 源码微调

- 针对目前lerobot分支需要进行如下的修改：

1、 在video_utils.py文件内部需要修改torchcodec为pyav，因为这是视频编码格式问题，torchcodec容易导致不稳定，我们无论哪种方案都选择使用稳定的pyav

![image-20250813094305264](./pic/image-20250813094305264.png)

2、在camera_opencv.py将多余的部分删除，保留self.index_or_path

![image-20250813094944219](./pic/image-20250813094944219.png)

3、在执行数据集录制的过程中会出现因为双臂通信存在延迟问题，导致主、从臂连接不上从而引发录制数据集num_try次数达到上限，所以修改num_try次数避免数据集录制中断。具体步骤为修改koch_follower中部分代码问题

![image-20250813095752843](./pic/image-20250813095752843.png)

## 利用脚本使用lerobot框架内部的有关koch机器臂的相关代码

- Koch机器臂标定部分代码使用

使用的是calibrate部分的代码，选择好相对应的主从机器臂。

```python
标定：
python -m lerobot.calibrate --teleop.type=koch_leader --teleop.port=COM7 --teleop.id=my_leader

python -m lerobot.calibrate --robot.type=koch_follower --robot.port=COM4 --robot.id=my_follower
```

- 录制数据集相关代码

使用record部分的代码，注意几个问题：端口名称、robot_id与之前标定的时候保持一致，相机的index，录制周期数以及录制时间

```python
录制数据集：

python -m lerobot.record     --robot.type=koch_follower     --robot.port=COM4     --robot.id=my_follower     --robot.cameras="{ left: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, color_mode: 'rgb', rotation: 'NO_ROTATION'}, above: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, color_mode: 'rgb', rotation: 'NO_ROTATION'}}"     --teleop.type=koch_leader     --teleop.port=COM7     --teleop.id=my_leader     --display_data=true     --dataset.repo_id=renjielv030/training_datasets     --dataset.num_episodes=100     --dataset.episode_time_s=15     --dataset.reset_time_s=10     --dataset.single_task="Pick up the rectangular block and put it into the cup."     --dataset.push_to_hub=False    --dataset.root=D:\dataset_for_smolvla    --resume=false
```

- 训练代码

这是针对koch机器臂的训练，采取方法为VA模型的ACT算法以及其变种。注意output的目录，是在运行程序的当前目录，或者直接设置其固定路径。

```python
python -m lerobot.scripts.train   --policy.type=act   --policy.push_to_hub=false  --dataset.root=H:/dataset_for_smolvla
 --dataset.repo_id=lizhuohang/block_cup_dataset_ACT   --batch_size=16   --steps=20000   --output_dir=outputs/train/my_koch_lzh   --job_name=my_kochACT_training --policy.device=cuda   --wandb.enable=false --policy.device=cuda
```

- 评估、真机演示代码

这是针对koch机器臂选择指定策略，如act策略训练之后得到的模型权重来进行真机演示操作。



## 数据集相关定义部分

## ACT模型

### pusht数据集

- meta数据集的内容

  1. episodes.jsons

     材料选取自record部分，包括录制数据集预训练以及

     ```json
     {"episode_index": 0, "tasks": ["Push the T-shaped block onto the T-shaped target."], "length": 161}
     {"episode_index": 1, "tasks": ["Push the T-shaped block onto the T-shaped target."], "length": 118}
     ```

     | 字段名        | 含义                                                         |
     | ------------- | ------------------------------------------------------------ |
     | episode_index | 当前记录这个周期的episode编号，这里记录的为第0个episode      |
     | tasks         | 任务描述，是一个自然语言指令，用于训练语言驱动的机器人控制（比如通过LLM理解任务） |
     | length        | 这个episode包含161帧/时间步，每一帧都有observation和action   |

- 