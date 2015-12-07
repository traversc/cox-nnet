library(ggplot2)
library(survival)

plotCoxMLP <- function(pred, time_test, event_test, png, title) {
    pred <- as.numeric(pred)
    split_r = as.numeric(pred > median(pred))
    diff = survdiff(formula = Surv(time_test,event_test) ~ split_r)
    pval = 1-pchisq(diff$chisq,1)

    high_curve<-survfit(formula = Surv(time_test[split_r == 1],event_test[split_r == 1]) ~ 1)  
    low_curve<-survfit(formula = Surv(time_test[split_r == 0],event_test[split_r == 0]) ~ 1) 

    png(png, width = 360, height = 360)
    plot(high_curve,main="", xlab="Time to event (days)", ylab="Probability", col= "blue", conf.int=F, lwd=1, xlim=c(0, max(time_test)))
    lines(low_curve, col= "green", lwd=1, lty=1, conf.int=F)
    legend("topright", legend=c("High PI", "Low PI"), fill=c("blue","green"))
    title(main=title, sub=paste0("pval = ", signif(pval,4)))
    dev.off()
    print(paste("MLP", pval))
}

## Plots for PBC
read.csv("PBC_cv_likelihoods.csv", header=F) -> cvll
L2 <- seq(-4.5,0.5,0.5)
se <- function(x) sd(x)/sqrt(length(x))
se <- apply(cvll, 1, se)
mean <- apply(cvll, 1, mean)
data <- data.frame(L2, mean, se)
png("PBC_cv_likelihoods.png", width = 360, height = 360)
g <- ggplot(data, aes(x=L2, y=mean)) + 
    geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.1) +
    geom_line() +
    geom_point() + labs(title = "PBC CV log likelihoods vs. L2 parameter")
print(g)
dev.off()

as.numeric(readLines("PBC_ystatus_test.csv")) -> event_test
as.numeric(readLines("PBC_ytime_test.csv")) -> time_test
readLines("PBC_theta.csv") -> pred
plotCoxMLP(pred, time_test, event_test, "PBC_survival_curves.png", "PBC median split")


## Plots for KIRC


read.csv("KIRC_cindex.csv", header=F) -> cvll
L2 <- seq(-3,1.67,0.33)

se <- function(x) sd(x)/sqrt(length(x))
se <- apply(cvll, 1, se)
mean <- apply(cvll, 1, mean)
data <- data.frame(L2, mean, se)
png("KIRC_cindex.png", width = 360, height = 360)
g <- ggplot(data, aes(x=L2, y=mean)) + 
    geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.1) +
    geom_line() +
    geom_point() + labs(title = "KIRC C-Index vs. L2 parameter")
print(g)
dev.off()

as.numeric(readLines("KIRC_ystatus_test.csv")) -> event_test
as.numeric(readLines("KIRC_ytime_test.csv")) -> time_test
readLines("KIRC_theta.csv") -> pred
plotCoxMLP(pred, time_test, event_test, "KIRC_survival_curves.png", "KIRC median split")