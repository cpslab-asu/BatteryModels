function [tout, yout] = run_Battery(~, u, T)
%     ts = u(:,1);
%     us = u(:,2:end);
%     disp(size(ts))
%     disp(size(us))
% 
%     tin = 0:0.1:T;
%     xin = interp1(ts, us, tin, 'previous');
%     u = [tin' xin];
%     disp(size(u))

    assignin('base','u',u);
    assignin('base','T',T);
    
    result = sim('test_fan_aircooling_V3.slx', ...
        'StopTime', 'T', ...
        'LoadExternalInput', 'on', 'ExternalInput', 'u', ...
        'SaveTime', 'on', 'TimeSaveName', 'tout', ...
        'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
        'SaveFormat', 'Array');
    tout = result.tout;
    yout = result.yout;
end