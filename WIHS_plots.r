library(survival)

read.csv("WIHS/x_train.csv", header=F) -> x_train
read.csv("WIHS/x_test.csv", header=F) -> x_test

as.numeric(readLines("WIHS/ystatus_train.csv")) -> event_train
as.numeric(readLines("WIHS/ystatus_test.csv")) -> event_test
as.numeric(readLines("WIHS/ytime_train.csv")) -> time_train
as.numeric(readLines("WIHS/ytime_test.csv")) -> time_test

readLines("theta.csv") ->pred

plotCoxMLP <- function(pred, time_test, event_test, png, title) {
    pred <- as.numeric(pred)
    split_r = as.numeric(pred > median(pred))
    diff = survdiff(formula = Surv(time_test,event_test) ~ split_r)
    pval = 1-pchisq(diff$chisq,1)

    high_curve<-survfit(formula = Surv(time_test[split_r == 1],event_test[split_r == 1]) ~ 1)  
    low_curve<-survfit(formula = Surv(time_test[split_r == 0],event_test[split_r == 0]) ~ 1) 

    png(png)
    plot(high_curve,main="", xlab="Time to event (days)", ylab="Probability", col= "blue", conf.int=F, lwd=1, xlim=c(0, max(time_test)))
    lines(low_curve, col= "green", lwd=1, lty=1, conf.int=F)
    legend("topright", legend=c("High PI", "Low PI"), fill=c("blue","green"))
    title(main=title, sub=paste0("pval = ", signif(pval,4)))
    dev.off()
    print(paste("MLP", pval))
}

plotCoxMLP(pred, time_test, event_test, "cox_mlp_WIHS.png", "Cox-nnet, WIHS dataset")