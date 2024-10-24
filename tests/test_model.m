clear;
t__ = [  0;900;1800;2700;3600;4000];
u__ = [-7 -7 -7 -7 -7 -7]';

u = [t__, u__];
T = 4000;
load("Batteries_v2.mat")
[tout, yout] = run_Battery([], u, T);



% G(0<SOC<1) and F(V<2)
% F(SOC<0) or F(SOC>1)