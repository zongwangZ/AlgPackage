// 自定义概率密度函数
functions {
  real[] getPrCorrect(real[] netStatus, real pr_network, int T, real[] prc) { // 获得共享路径 m2 正确测量的概率
    real pr[T];
    
    for (t in 1:T) {
      if (netStatus[t] <= pr_network) { // 网络处于适宜状态
        pr[t] = prc[1];
      } else {                  // 网络处于非适宜状态
        pr[t] = prc[2];
      }
    }

    return (pr);
  }
  
  real[,,] getThetaM2(vector sharedPathLength, real[] p_correct, int N, int T) { // 获得 m2 的概率分布
    real theta_m2[T, N*(N-1)/2, N-1];
    real l_true;
  
    for (t in 1 : T) {
      for (n in 1 : N*(N-1)/2) {
        l_true = round(sharedPathLength[n]);
        for (l in 1 : (N-1)) {
          if (l <= l_true && l > (l_true-1)) {
            theta_m2[t,n,l] = p_correct[t];
          } else {
            theta_m2[t,n,l] = (1 - p_correct[t]) / (N-2);
          }
        }
      }
    }
  
    return (theta_m2);
  }
  
  real[,,] getThetaM3 (real[,,] theta_m2, int[,] index_m2, int[,,] index_m3, int N, int T) { // 获得 m3 的概率分布
    real theta_m3[T, N*(N-1)*(N-2)/6, 5];
    
    for (t in 1:T) {
      for (i in 1:(N-2)) { // 获得 m3 的概率分布
      for (j in (i+1):(N-1)) {
        for (k in (j+1):N) {
          int ijk = index_m3[i,j,k];
          int ij = index_m2[i,j];
          int ik = index_m2[i,k];
          int jk = index_m2[j,k];
          
          theta_m3[t, ijk, ] = rep_array(0,5);
          for (l in 1:(N-1)) {
            theta_m3[t, ijk, 1] += theta_m2[t, ij, l] * theta_m2[t, ik, l] * theta_m2[t, jk, l];
            
            if (l >= 2) {
              for (l_less in 1:(l-1)) {
                theta_m3[t, ijk, 2] += theta_m2[t, ij, l] * theta_m2[t, ik, l_less] * theta_m2[t, jk, l_less];
                theta_m3[t, ijk, 3] += theta_m2[t, ij, l_less] * theta_m2[t, ik, l] * theta_m2[t, jk, l_less];
                theta_m3[t, ijk, 4] += theta_m2[t, ij, l_less] * theta_m2[t, ik, l_less] * theta_m2[t, jk, l];
              }
            }
          }
          
          theta_m3[t, ijk, 5] = 1;
          for (topo in 1:4) { // m3测量失败的概率
          theta_m3[t, ijk, 5] += -1 * theta_m3[t, ijk, topo];
          }
          
        }
      }
    }
  }
  
  return (theta_m3);
  }
  
  real measure_m2_lpmf (int y, real[] theta, int num_u) { // m2 概率分布
    real prob = 0.0;
    for (i in 1:num_u) {
      if (y == i) {
        prob = theta[i];
      }
    }
//    if(prob == 0)
//    {
//        prob = 0.000001;
//    }
//    else if(prob == 1)
//    {
//        prob = 0.999999;
//    }
    return (log(prob));
  }
    
  real measure_m3_lpmf (int y, real[] theta) { // m3 概率分布
    real prob = 0.0;
    for (topo in 0:4) {
      if (y == topo) {
        prob = theta[topo+1];
      }
    }
//    if(prob == 0)
//    {
//        prob = 0.000001;
//    }
//    else if(prob == 1)
//    {
//        prob = 0.999999;
//    }
    return (log(prob)); 
  }
}

// 输入的数据为：端到端测量路径数目N，测量周期数目T，
// 以及三路测量结果 m3_observed
data {
  int<lower=0> do_debug; // 指示是否打印中间变量

  int<lower=4> N;
  int<lower=1> T;
  int<lower=0, upper=4> m3_observed[N*(N-1)*(N-2)/6, T];
  // int<lower=1> m2_observed[N*(N-1)/2, T];
  real<lower=0, upper=1> prc[2];
  
  int<lower=0> index_m2[N, N];     // 共享路径 m2 编号索引
  int<lower=0> index_m3[N, N, N];  // 三路子拓扑 m3 编号索引

  real<lower=0, upper=1> net_status[T]; //将parameters中带估计的参数移到data中，减少推断的难度
  real<lower=0, upper=1> r_ns; //先设置为1，减少难度
}

// 未知的参数为：共享路径长度m2，网络状态net_status，
// 网络状态正常的概率r
parameters {
  vector<lower=0.5, upper=(N-0.5)>[N*(N-1)/2] m2;

  // real<lower=0, upper=1> net_status[T]; //注释掉减少推断难度
  // real<lower=0, upper=1> r_ns;           //注释掉减少推断难度
}

// 指定先验分布以及似然函数。不考察的变量与参数放到model里可以降低计算量和节约存储空间
model {
  real theta_m2[T, N*(N-1)/2, N-1];       // 共享路径 m2 准确性的离散概率分布
  real theta_m3[T, N*(N-1)*(N-2)/6, 5];   // 三路子拓扑结构 m3 \in {0, 1, 2, 3, 4} 的
                                          // 测量准确性的离散概率分布，其中
                                          // 4 代表测量错误结果
  real p[T];                              // 正确测量共享路径的概率
  
  m2 ~ uniform(0.5, N-0.5);              // 共享路径 m2 的先验长度

  // r_ns ~ uniform(0.8, 1);                // 网络处于适宜状态的先验概率
                                            //注释掉，减少推断难度
  // net_status ~ uniform(0, 1);            // 先验网络状态；在每个测量时隙中，网络状态不变
                                            //注释掉，减少推断难度
  
  if (do_debug) {
    print("m2=", m2);
    print("r_ns=", r_ns);
  }
  
  p = getPrCorrect(net_status, r_ns, T, prc);                  // 获得共享路径 m2 正确测量的概率
  theta_m2 = getThetaM2(m2, p, N, T);                          // 获得共享路径 m2 不同测量结果的离散概率分布
  theta_m3 = getThetaM3(theta_m2, index_m2, index_m3, N, T);   // 获得三路子拓扑 m3 不同测量结果的离散概率分布
  for (t in 1:T) {  // 在每个测量时隙中，对所有的三路子拓扑进行采样
    for (i in 1:(N*(N-1)*(N-2)/6)) {
          m3_observed[i, t] ~ measure_m3(theta_m3[t, i, ]);       // 对三路子拓扑 m3 进行采样
    }

    // for (i in 1:(N*(N-1)/2)) {
      //     m2_observed[i, t] ~ measure_m2(theta_m2[t, i, ], N-1);  // 对共享路径 m2 进行采样
    // }
    
    if (do_debug) {
      print("t=", t);
      print("net_status=", net_status[t]);
      print("p=", p[t]);
      print("theta_m2=", theta_m2[t]);
      print("theta_m3=", theta_m3[t]);
      print("sample_m3=", m3_observed[,t]);
    }
  }
}
