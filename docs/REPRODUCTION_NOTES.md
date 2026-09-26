# 论文协议与未明确项

## 严格按正文/附录实现

- DiT-XL/2，256×256，DDPM 50 次执行步，CFG=4；λ=10，tc=795。
- `SpacedDiffusion` 将采样索引映射到原始 0..999 后再调用 UAG。不是对 50 步索引应用 795。
- Eq.17 先计算 CFG epsilon 并 detach，然后从 latent noisy state 构建 clean estimate，经 VAE decode、224 resize 和 ImageNet normalize 后求熵。仅冻结参数，不能用 no_grad 包住 VAE/分类器的引导路径。
- `entropy.sum()` 保持每张图的 λ 不随 batch 大小改变；不加熵阈值、不做梯度裁剪、不夹紧 clean estimate。
- 原始 DiT 的 `forward_with_cfg` 默认只对前三个 latent 通道做 CFG；保持原实现。UAG 修正全部四个 epsilon 通道，最后四个 learned-variance 通道不变。
- 固定每张图 seed，配对版本共享初始噪声与全部 DDPM 随机噪声序列。保存每张图的 RNG 哈希与 checkpoint 来源。
- 参考 ResNet-18：子集从头训练 200 轮，SGD(.1,momentum=.9,wd=1e-4)，batch 256，cosine；ImageNet-1K 使用预训练 ResNet-18。
- 硬标签学生：AdamW(lr=.001, betas=.9/.999, wd=.01)，2/3、5/6 时 lr×.2；IPC 1/10=2000 轮、20/50=1500 轮、70/100=1000 轮。
- 学生 ConvNet-6、ResNetAP-10、ResNet-18 复用 Minimax 的 instance-normalized 定义（使用等价的逐通道 GroupNorm）；标准参考 ResNet-18 使用 BN。
- 效用为原式的 gradient-flow removal，不用梯度范数、熵或影响函数替代。B 为均值损失时，差值可化成 `g_j·(g_i-g_mean)/(B-1)`，取 j≠i 的绝对最大值。
- 25/50/75% 学生训练位置保存 checkpoint；每个样本每 checkpoint 5 个 batch-size-64 上下文，诊断不更新模型参数。
- 512 维、L2-normalized 的预训练 ResNet18 特征；只在真实训练集上聚类，所有生成方法共享 centers；各 K 独立固定 seed。
- 10000 次联合置换：每个 K、每个 class 内置换，统计量为四种 K 的 Spearman 均值，两侧检验。
- 三个独立 generation/student seeds=0/1/2。结果默认统计 final epoch top1，另记 best，不以 best 替换 final。

## 作者没有提供、因此明确固定的工程选择

1. **主表 checkpoint**：稿件说“where applicable 使用 Minimax checkpoint”，但未给逐行映射。默认 `--main-checkpoint dit`；另提供 minimax 选项。不能保证某一默认配置对应主表某行。
2. **VAE/CFG**：正文未指定 VAE 版本和 CFG 通道。独立采样使用mse VAE 和官方三通道 CFG；插件使用各自上游默认 VAE。所有选择可追踪，不能声称与作者完全一致。
3. **子集类别**：Woof/Nette/100 来自固定 Minimax 仓库；IDC 使用常用的 green mamba、garden spider、Saluki、Doberman、langur、bonnet、cocktail shaker、vacuum、window screen、gyromitra 十类。论文未给 WNID 列表，作者释放后应核对；不会误用 class100 的前十类冒充 IDC。
4. **硬标签增强**：batch 64，224 预处理，RRC scale=(.5,1)，horizontal flip，CutMix Beta(1,1)、概率1、按实际区域比例重算标签权重。稿件没有提供这些参数。未宣称增强细节被完全还原。
5. **软标签**：默认每图均分 2×2 区域，保存区域和 teacher 概率；训练默认对同一增强后区域在线重新计算 teacher targets，避免 RandAugment 变换后标签错位。可显式 `--fixed-soft-targets` 使用缓存。正文未给区域数、label 温度和增强参数，默认温度1、RandAugment N=2/M=9、batch256；scheduler 沿用硬标签 milestones。这个流程是完整可执行的明确实现，而不是作者未提供的逐增强缓存协议的精确还原。
6. **WRN**：提供标准三阶段预激活 WRN-28-10，224 输入，不添加未经说明的 stride stem。正文未给其训练/预处理细节；统一参考训练器，可调整 batch 以适应实际显存，但应记录协议变化。ViT 使用 timm ViT-Tiny/16。未实际测量显存。
7. **空聚类区域**：没有生成样本的区域，其均值效用未定义，保留 NaN，并按相关的有效 pair 排除；不把效用填0。置换仅在有效 pair 内进行。所有报告记录覆盖率。论文未说明空区域处理。
8. **冗余诊断**：每类分成5份，每份独立随机初始化，默认 ResNet-18、根据子集实际 IPC 排训练轮数，固定 eval 预处理。论文未指定学生架构/训练超参/分层方式；参数可显式调整，不能保证重现 Table7 极大的 gap。
9. **效用诊断**：默认 ImageWoof IPC50/ResNet18（论文未给这些诊断的完整配置），checkpoint eval 模式和固定验证式预处理，不加入随机增强干扰上下文。共享上下文索引不意味着 base/UAG 的 checkpoint 参数相同；各自在自己的合成集上训练。
10. **插件 MGD3**：官方实现用 posterior-mean 上的 mode correction。本适配在 CFG epsilon 输出处加入 UAG，然后继续原有 mode correction。论文描述了泛化 epsilon 叠加，却没给 MGD3 的精确插入代码，因此记录这一选择，不宣称作者级完全同构。
11. **Minimax/IMS3 微调**：复用上游算法，预设8轮、batch8，每1000步保存以适配9025张训练图片的 ImageWoof 子集。与某些上游脚本12000步保存相比，仅改变保存频率以免短训练没有任何 checkpoint。默认链接最后保存 checkpoint，可用 `--step` 锁定。没有自动训练、没有凭空创建微调权重。
12. **时间统计**：单图 GPU 结果取回 CPU 后记录采样耗时；重启后的 complete.json 只代表本次调用耗时。完整机器耗时可用 `/usr/bin/time -v` 包裹实际任务。计算成本应来自实际运行的计时记录。

