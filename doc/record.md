m3.stan文件中的prc和net_status可以从参数block中移除，由data中给出
这样的话，可以减少计算的复杂度。这时候，参数只剩下m2，但是出现了
Rejecting initial value:
  Log probability evaluates to log(0), i.e. negative infinity.
  Stan can't start sampling from this initial value.
Initialization between (-2, 2) failed after 1 attempts. 
 Try specifying initial values, reducing ranges of constrained values, or reparameterizing the model.
 的错误信息，估计是计算log概率时候，值为0，log后为无穷了，因此接下来需要关注那
 两个自定义的概率函数的逻辑，同时思考为什么之前出现这种问题后，依然可以继续运行。
 r_ns ~ uniform(1, 1)会初始化失败 
 
 得出的结果和初始化关系非常大，尝试扩大初始化范围