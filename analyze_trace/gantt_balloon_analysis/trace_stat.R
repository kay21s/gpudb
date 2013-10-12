###################################################################
#                  Query Trace Statistics on GPU
###################################################################

library(gplots)


sq_box = function(filename, reader=file2array2) {
    ret <- reader(filename)

    T <- ret[[1]]
    etypes_id <- ret[[2]]
    etypes <- ret[[3]]

    par(mar=c(7,5,1,1))
    boxplot(c(10),
        xlim = c(0,length(etypes_id)+1), boxwex=0, log="y",
        col="white", varwidth=TRUE, 
        main="Query Operation Statistics",
        ylab="Completion Time", ylim=c(max(min(T[,"duration"]), 1), max(T[,"duration"])))
    axis(side = 1, labels = FALSE)
    title(xlab="Operation Type",mgp=c(6,1,0))
    text(1:length(etypes), 0.25, labels = etypes, srt = 45, pos = 1, xpd = TRUE, cex=0.8)

    pos <- 1
    len1 <- length(T[T["event"]==etypes_id[1]])
    color <- heat.colors(length(etypes_id));

    for (id in etypes_id) {
        d = T[T["event"]==id, "duration"]
        scale = sqrt(length(d)/len1)/5      

        boxplot(d, add = TRUE, varwidth=TRUE, log="y",
            boxwex = scale, at = pos, col=color[pos])

        pos <- pos + 1
    }
}


kq_box = function(files, reader = file2array2) {

    minT <- Inf
    maxT <- -Inf
    for (file in files) {
        ret <- reader(file)
        T <- ret[[1]]
        etypes_id <- ret[[2]]
        etypes <- ret[[3]]
        minT <- min(min(T[,"duration"]), minT)
        maxT <- max(max(T[,"duration"]), maxT)
    }

    par(mar=c(7,5,1,1))
    boxplot(c(10),
        xlim=c(0,length(etypes_id)+1), boxwex=0, log="y",
        col="white", varwidth=TRUE, 
        main="Query Operation Statistics",
        ylab="Completion Time", ylim=c(max(minT, 1), maxT))
    axis(side = 1, labels = FALSE)
    title(xlab="Operation Type", mgp=c(6,1,0))
    text(1:length(etypes), 0.25, labels=etypes, srt = 45, pos = 1, xpd = TRUE, cex=0.8)

    len1 <- length(T[T["event"]==etypes_id[1]])

    fnum <- length(files)
    offset <- 1/fnum
    m <- (fnum+1)/2
    color <- heat.colors(fnum)
    fid <- 1

    for (file in files) {
        T <- reader(file)[[1]]
        pos <- 1
        for (id in etypes_id) {
            d = T[T["event"]==id, "duration"]
            scale = sqrt(length(d)/len1)/(5*fnum)

            boxplot(d, add = TRUE, varwidth=TRUE, log="y",
                boxwex = scale, at = pos + offset*(fid - m), col=color[fid])

            pos <- pos + 1
        }
        fid <- fid + 1
    }

    smartlegend(x="left",y="top", inset=0, cex=0.8,
        files, fill=color, ncol=2)
}



kq_balloon = function(files, reader = file2array2) {

    ret <- reader(files[1])
    etypes_id <- ret[[2]]
    etypes <- ret[[3]]
    queries <- files    

    datavals <- matrix(1, ncol=length(etypes), nrow=length(queries))
    fid <- 1
    for (file in files) {
        T <- reader(file)[[1]]
        pos <- 1
        for (id in etypes_id) {
            d = T[T["event"]==id, "duration"]
            datavals[fid, pos] <- sum(d)
            pos <- pos + 1
        }
        fid <- fid + 1
    }


    data <- data.frame(Query=rep(queries,length(etypes)),
        Color=rep(etypes, rep(length(queries), length(etypes)) ),
        Value=as.vector(datavals) )
    levels(data$Query) <- queries   

    balloonplot( data$Query, data$Color, data$Value, ylab ="Operation", xlab="Query", 
        main="Completion Time Statistics of Query Operations on GPU")

}

sq_gantt = function(file, reader = file2array2) {

    ret <- reader(file)
    T <- ret[[1]]
    etypes_id <- ret[[2]]
    etypes <- ret[[3]]
    colors <- topo.colors(length(etypes))

    labels <- as.character(T[,1])
    for (i in 1:length(etypes)) {
        labels <- replace(labels, labels == (i-1), etypes[i])   
    }   

    Ymd.format <- "%Y"
    Ymd <- function(x){ as.POSIXct(x, origin="0000-01-01")}
    gantt.info <- list(
        labels     =labels,
        starts     =Ymd(T[,2]),
        ends       =Ymd(T[,2]+T[,3]),
        priorities = T[,1]+1)
    gantt.chart(gantt.info,main=file, vgrid.format="%Y", taskcolors=colors)
}

kq_gantt = function(files, reader = file2array2) {

    ret <- reader(files[[1]])
    T <- ret[[1]]
    etypes_id <- ret[[2]]
    etypes <- ret[[3]]
    colors <- topo.colors(length(etypes))

    crs <- c()
    cre <- c()
    crl <- c()
    crp <- c()

    for (file in files) {
        T <- reader(file)[[1]]

        rs <- T[,2]
        re <- T[,2] + T[,3]
        rp <- T[,1]
        rl <- rep(file, length(rs))

        crs <- append(crs, rs)
        cre <- append(cre, re)
        crl <- append(crl, rl)
        crp <- append(crp, rp)
    }

    Ymd <- function(x){ as.POSIXct(x, origin="0000-01-01")}
    gantt.info <- list(
        labels     =crl,
        starts     =Ymd(crs),
        ends       =Ymd(cre),
        priorities = crp+1)
    gantt.chart(gantt.info,main="Query Execution", vgrid.format="%Y", taskcolors=colors, vgridlab=FALSE)
    smartlegend(x="right",y="top", inset=0, cex=0.8,
        etypes, fill=colors, ncol=1)
}

box_prep = function(){
    setwd("/Users/kay21s/projects/R")
    library("plotrix")
    source("trace_prep.R")
    files <- dir("./anal/")
    setwd("./anal/")
    kq_gantt(files)
}

balloon_prep = function(){
    setwd("/Users/kay21s/projects/R")
    library("plotrix")
    source("trace_prep.R")
    files <- dir("./anal/")
    setwd("./anal/")
    kq_balloon(files)
}
