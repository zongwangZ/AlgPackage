首先要验证，在输入数据完备的状况下，该算法可以准确推断
经过实验，发现目的无法取到最优，估计还是因为m2的分布为均匀分布，m2测量概率用了
round，导致采样过程优化失败。
主要原因估计还是因为训练不够，查找发现，暂时无法让参数m2服从离散均匀分布。