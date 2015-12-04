using PyPlot

RMSEtest500 = h5read("res_batchsize500.h5","RMSEtest500")
RMSEtrain500 = h5read("res_batchsize500.h5","RMSEtrain500")
timer_learn500 = h5read("res_batchsize500.h5","timer_learn500")
timer_test500 = h5read("res_batchsize500.h5","timer_test500")
RMSEtrainve500 = h5read("res_batchsize500.h5","RMSEtrainve500")
RMSEtestvec500 = h5read("res_batchsize500.h5","RMSEtestvec500")

RMSEtest50 = h5read("res_batchsize50.h5","RMSEtest50")
RMSEtrain50 = h5read("res_batchsize50.h5","RMSEtrain50")
timer_learn50 = h5read("res_batchsize50.h5","timer_learn50")
timer_test50 = h5read("res_batchsize50.h5","timer_test50")
RMSEtrainve50 = h5read("res_batchsize50.h5","RMSEtrainve50")
RMSEtestvec50 = h5read("res_batchsize50.h5","RMSEtestvec50")

RMSEtest5000 = h5read("res_batchsize5000.h5","RMSEtest5000")
RMSEtrain5000 = h5read("res_batchsize5000.h5","RMSEtrain5000")
timer_learn5000 = h5read("res_batchsize5000.h5","timer_learn5000")
timer_test5000 = h5read("res_batchsize5000.h5","timer_test5000")
RMSEtrainve5000 = h5read("res_batchsize5000.h5","RMSEtrainve5000")
RMSEtestvec5000 = h5read("res_batchsize5000.h5","RMSEtestvec5000")


timervec5000 = linspace(0,(timer_learn5000 + timer_test5000),length(RMSEtestvec5000))
timervec500 = linspace(0,(timer_learn500 + timer_test500),length(RMSEtestvec500))
timervec50 = linspace(0,(timer_learn50 + timer_test50),length(RMSEtestvec50))

plot(collect(timervec500), collect(RMSEtestvec500), label = "m = 500")
plot(collect(timervec50), collect(RMSEtestvec50),label ="m = 50")
plot(collect(timervec5000), collect(RMSEtestvec5000), label = "m = 5000")
