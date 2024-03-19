clc; clear; close all;


%% 초기값 설정
t = 1; % 측정 간격
n = 9000; % 측정 횟수

%% Target initalize
target_heading =deg2rad(30); % deg
vel_target=20;
rotation_target=deg2rad(20);
initial_target_State = [400 0 vel_target*cos(target_heading) vel_target*sin(target_heading)]'; % x y x' y'
%% Sensor initaialize
sensor1_heading =deg2rad(60); % deg
sensor2_heading =deg2rad(10); % deg
rotation_sensor=[deg2rad(90); deg2rad(-20)];%angle to change for sensor1, sensor2 

vel_sensor1=17;
vel_sensor2=21;
initial_sensor1_State = [0 0 vel_sensor1*cos(sensor1_heading) vel_sensor1*sin(sensor1_heading)]'; % sensor1_x sensor1_y sensor1_x' sensor1_y'
initial_sensor2_State = [1000 0 vel_sensor2*cos(sensor2_heading) vel_sensor2*sin(sensor2_heading)]'; % sensor2_x sensor2_y sensor2_x' sensor2_y'
G_noise=[1;1];

%% EKF Setting
P = eye(4); % 오차 공분산 행렬
Q = 10*eye(4); % 시스템 잡음 공분산 행렬
R = 0.1*eye(2); % 측정 잡음 공분산 행렬
target_model = [1 0 1 0;%model
    0 1 0 1;
    0 0 1 0;
    0 0 0 1];
sensor_model = [1 0 1 0;%model
    0 1 0 1;
    0 0 1 0;
    0 0 0 1];
%% Data initializing
true_trajectory = zeros(n, 2);
estimated_trajectory = zeros(n, 4);
sensor1_trajectory = zeros(n, 2);
sensor2_trajectory = zeros(n, 2);

target_state = initial_target_State;
X=target_state;
sensor1_state = initial_sensor1_State;
sensor2_state = initial_sensor2_State;
error=zeros(n,5);

%% Main
for i = 1:n
    % Prediction
    time=i*t;
    if time==n/2
        sensor1_state=rotate_model(sensor1_state,vel_sensor1,sensor1_heading,rotation_sensor(1));
        sensor2_state=rotate_model(sensor2_state,vel_sensor2,sensor2_heading,rotation_sensor(2));
        target_state=rotate_model(target_state,vel_target,target_heading,rotation_target);
        X=target_state+[e(1); e(2); e(3); e(4)];
    end
    target_state = motion_model(target_state,target_model);
    sensor1_state=motion_model(sensor1_state,sensor_model);
    sensor2_state=motion_model(sensor2_state,sensor_model);
    
    true_trajectory(i, :) = target_state(1:2);
    sensor1_trajectory(i, :) = sensor1_state(1:2);
    sensor2_trajectory(i, :) = sensor2_state(1:2);
    X=target_model*X;

    % 야코비안 계산
    H = measurement_model(X, sensor1_state, sensor2_state);
    %오차 공분산 예측
    P = prediction_step(P, Q);
    
    % 칼만 이득 계산
    K = P * H' * inv(H * P * H' + R);
    G_noise(1)=norm(true_trajectory(i,1)-sensor1_trajectory(i,1),true_trajectory(i,2)-sensor1_trajectory(i,2));
    G_noise(2)=norm(true_trajectory(i,1)-sensor2_trajectory(i,1),true_trajectory(i,2)-sensor2_trajectory(i,2));
    G_noise=G_noise/(4*n);
    % 실제 경로의 상대 방위각 정보(참값)
    z = h(target_state,sensor1_state,sensor2_state);
    % 참값에 노이즈 추가->센서값
    z = z + [deg2rad(G_noise(1)*randn());deg2rad(G_noise(2)*randn())];
    %비선형 측정 방정식h(X)
    X = X + K * (z - h(X,sensor1_state,sensor2_state));
    P = (eye(4) - K * H) * P;
    DP=diag(P);
    e(i,5)=DP(2);
    disp(e(i,5));
    % save result
    estimated_trajectory(i, :) = X(1:4);
    er=X-target_state;
    e(1:4)=er(1:4);
    error(i,:)=[e(1);e(2); e(3);e(4);e(5)];
end

%% Plot
figure;
plot(true_trajectory(:, 1), true_trajectory(:, 2), 'k*', 'LineWidth', 2); hold on;
plot(estimated_trajectory(:, 1), estimated_trajectory(:, 2), 'g.', 'LineWidth', 2);
plot(sensor1_trajectory(:, 1), sensor1_trajectory(:, 2), 'r-', 'LineWidth', 2);
plot(sensor2_trajectory(:, 1), sensor2_trajectory(:, 2), 'r-', 'LineWidth', 2);

xlabel('X 위치');
ylabel('Y 위치');
title('타겟의 궤적과 추정 궤적(m)'); 
legend('실제 궤적', '추정 궤적', '센서 위치');

figure;

plot(1:n, error(:, 1), 'm', 'LineWidth', 2);hold on;
plot(1:n, error(:, 2), 'y', 'LineWidth', 2);
xlabel('시간');
ylabel('거리 오차');
title('타겟 위치 추정의 거리 오차(m)');
ylim([0 1000]);
legend('Position Error_x', 'Position Error_y');

figure;
plot(1:n, error(:, 3), 'm', 'LineWidth', 2);hold on;
plot(1:n, error(:, 4), 'y', 'LineWidth', 2);
xlabel('시간');
ylabel('속도 오차');
title('타겟 위치 추정의 속도 오차(m/s)');
legend('Velocity Error_x', 'Velocity Error_y');

figure;
plot(1:n, error(:, 5), 'm', 'LineWidth', 2);hold on;
xlabel('시간');
ylabel('P');
title('P');

%% Function
% physical model
function x = motion_model(x,f)
    x=f*x;
end
%change rotation
function x=rotate_model(x,vel,prev_angle,update_angle)
    x=x+[0;0; vel*cos(prev_angle-update_angle); vel*sin(prev_angle-update_angle)];
end

% measurement_model
function H = measurement_model(target_state, sensor1_state, sensor2_state)
    dx1 = target_state(1) - sensor1_state(1);
    dy1 = target_state(2) - sensor1_state(2);
    dx2 = target_state(1) - sensor2_state(1);
    dy2 = target_state(2) - sensor2_state(2);
    q1 = dx1^2 + dy1^2;
    q2 = dx2^2 + dy2^2;
    H = [-dy1/(q1) dx1/(q1) 0 0;
         -dy2/(q2) dx2/(q2) 0 0];
end

%타겟의 XY 위치 정보를 베어링 정보로 변환(센서 데이터화(Type: bearing)
function z = h(target_state, sensor1_state,sensor2_state)
    dx1 = target_state(1) - sensor1_state(1);
    dy1 = target_state(2) - sensor1_state(2);
    dx2 = target_state(1) - sensor2_state(1);
    dy2 = target_state(2) - sensor2_state(2);
    
    z = [atan2(dy1, dx1);atan2(dy2,dx2)];
end

% P_prediction
function P = prediction_step(P,Q)
H = [1 0 1 0;
    0 1 0 1;
    0 0 1 0;
    0 0 0 1];
P = H*P*H' + Q;
end
