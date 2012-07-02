%%%% BUILD DATASET %%%%
clear;
training_mode = 1;
nb_samples = 1000; input_dim = 1; output_dim = 2; nb_topics = 3; nb_iter = 15;
x = -1 + 2*rand(nb_samples, input_dim+1);
x(:,end) = 1;

T = 2*(-1+2*rand(input_dim+1, nb_topics))/sqrt(input_dim);%/sqrt(output_dim);
% T = [-2, 0, 2;-1, 3, 1];
t = softmax((x*T)')';

experts = (-1+2*rand(input_dim + 1, output_dim, nb_topics))/sqrt(input_dim);%/sqrt(output_dim);
% experts(:, 1, :) = repmat([1,0]', 1, nb_topics);
% experts(:,2,1) = [-1, -4];
% experts(:,2,2) = [1, 0];
% experts(:,2,3) = [-1, 2];
y = mnrnd(1,t);

data = zeros(nb_samples, output_dim);
for i=1:nb_samples
    data(i,:) = x(i,:)*experts(:,:,find(y(i,:)==1)) + mvnrnd(zeros(1,output_dim), 0.05*eye(output_dim));
end

real_T = T;
real_expert = experts;

%%%% TRAIN WITH EM %%%%

%Initialization
T = (-1+2*rand(input_dim+1, nb_topics))/sqrt(input_dim)/sqrt(output_dim);
experts = (-1+2*rand(input_dim + 1, output_dim, nb_topics))/sqrt(input_dim)/sqrt(output_dim);

%Likelihod and subparts
lli = zeros(nb_samples, nb_topics);
Q1 = zeros(1, nb_iter);
Q2 = zeros(1, nb_iter);

%Training
for iter=1:nb_iter
    
    %%%% EXPECTATION %%%%
    gating_values = softmax((x*T)')';
    prediction_experts = zeros(nb_samples, nb_topics);
    
    for j=1:nb_topics
        prediction_experts(:,j) = mvnpdf(data, x*experts(:,:,j),  0.1*eye(output_dim)) + eps;
    end
    
    expectation = gating_values.*prediction_experts;
    expectation = expectation./repmat(sum(expectation, 2),1,nb_topics);
    
    %%%% MAXIMIZATION %%%%
    
    %Expert part likelihood
    for j=1:nb_topics
        lli(:,j) = mvnpdf(data, x*experts(:,:,j),  0.05*eye(output_dim)) + eps;
    end
    Q1(1, iter) = sum(sum(expectation.*log(lli)), 2);
    
    %Gate part likelihood
    Q2(1, iter) = sum(sum(expectation.*log(softmax((x*T)')'), 2));
    
    %Global Likelihood
    ll(1, iter) = sum(log(sum(gating_values.*prediction_experts,2))); 
    
    %Reweighted Linear Regression
    for i=1:nb_topics
%         experts(:,:,i) = lscov(x, data, expectation(:,i));
        w = diag(expectation(:,i));
        experts(:,:,i) = inv(x'*w*x)*x'*w*data;
    end
    
    
    %Verif RLS
    for j=1:nb_topics
        lli(:,j) = mvnpdf(data, x*experts(:,:,j),  0.05*eye(output_dim)) + eps;
    end
    if sum(sum(expectation.*log(lli)), 2) < Q1(1,iter)
        warning('Bad RLS at iter %d. Previous: %s, Actual: %s', iter, Q1(1,iter), sum(sum(expectation.*log(lli)), 2));
    end
    
    %Newton-Raphson
    if training_mode == 1
        input_dim = input_dim + 1;
        pe = zeros(nb_samples, nb_topics);
        hessian = zeros(input_dim*nb_topics, input_dim*nb_topics);
        stepsize = 1;
        
        %Gradient
        grad = x'*(softmax((x*T)')' - expectation)
        grad = reshape(grad, input_dim*nb_topics, 1);

        %Hessian
        prob = softmax((x*T)')';
        for topic=1:nb_topics
            w_diag = prob(:, topic).*(1-prob(:, topic));
            hessian((topic-1)*input_dim+1:topic*input_dim, (topic-1)*input_dim+1:topic*input_dim) = x'*diag(w_diag)*x;
            for other_topic=topic+1:nb_topics
                w_off_diag = -prob(:, topic).*prob(:,other_topic);
                hessian_off_diag = x'*diag(w_off_diag)*x;
                hessian((topic-1)*input_dim+1:topic*input_dim, (other_topic-1)*input_dim+1:other_topic*input_dim) = hessian_off_diag;
                hessian((other_topic-1)*input_dim+1:other_topic*input_dim, (topic-1)*input_dim+1:topic*input_dim) = hessian_off_diag;
            end
        end

        %Levenberg-Marquardt
        if rcond(hessian) < eps,
            for i = -16:16,
                h2 = hessian.*(( 1+ 10^i)*eye(size(hessian))  + (1-eye(size(hessian))));
                if rcond(h2) > eps, break, end
            end
        hessian = h2;    
        end

        delta = reshape(hessian\grad, input_dim, nb_topics);
        
        %Update Gates
        while stepsize > 1e-2
            T2 = T - stepsize*delta;

            if sum(sum(expectation.*log(softmax((x*T2)')'), 2)) > Q2(1,iter), break, end;
            stepsize = stepsize/2;
        end 
        T = T2;

        %Verif newton part
        if sum(sum(expectation.*log(softmax((x*T)')'), 2)) < Q2(1,iter)
            warning('Bad Newton-Raphson. Stepsize: %f', stepsize);
        end
        input_dim = input_dim - 1;
    end %end Newton Raphson
    
    %Gradient Ascent
    if training_mode == 2
        for gd_iter=1:200
            T = T - 0.00005*(x'*(softmax((x*T)')' - expectation));
        end
    end
end

%%%% PLOTTING %%%%
clf;
subplot(2,2,1);
d1 = x*experts(:,:,1);
d2 = x*experts(:,:,2);
d3 = x*experts(:,:,3);
plot(data(:,1),data(:,2),'kx', d1(:,1), d1(:,2), 'bx', d2(:,1), d2(:,2), 'rx', d3(:,1), d3(:,2), 'gx')

subplot(2,2,2);

tt = softmax((x*T)')';

hold on;
cmap= hsv(nb_topics);
for i = 1:nb_topics
    plot(x(:,1), t(:, i), 'kx')
    plot(x(:, 1), tt(:, i), 'x', 'color', cmap(i,:))
end
 
subplot(2,2,3)
% title('income')
% plot(1:nb_iter, Q1)
data2 = zeros(nb_samples, output_dim);
tt = softmax((x*T)')';
y = mnrnd(1,tt);
for i=1:nb_samples
    data2(i,:) = x(i,:)*experts(:,:,find(y(i,:)==1)) + mvnrnd(zeros(1,output_dim), 0.05*eye(output_dim));
end   
plot( data(:,1),data(:,2), 'k.', data2(:,1),data2(:,2), 'b.') 
 
subplot(2,2,4);
plot(1:nb_iter, ll)

   
