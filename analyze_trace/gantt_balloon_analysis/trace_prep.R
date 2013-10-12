###################################################################
#                  Query Trace Preprocessing
###################################################################

file2array1 = function(filename) {
	out <- read.table(filename, skip=1, colClasses="numeric")
	data <- out
	data[,3] = out[,3] - out[,2]
	names(data) <- c("event", "start", "duration")

	return(list(data,
			c(0,1,2,3,4,5),
			c("host-to-device memcpy","device-to-host memcpy",
				"GPU kernel execution", "diskIO",
                "cudaMalloc","CPU Operation") ))
}

file2array2 = function(filename) {
	out <- read.table(filename, skip=10, colClasses="numeric")
	data <- out
	data[,3] = out[,3] - out[,2]

	enum <- dim(data)[1]
	ecol <- 1:3
	eenum<- 2*enum -1
	edata <- data 
	edata[eenum,3] <- 0

	#add "event 4" records
	# event 4, end, duration: current start -last end
	edata[seq(1,eenum,2),] <- data
	edata[seq(2,eenum,2),1] <- 4
	edata[seq(2,eenum,2),2] <- head(data, -1)[,2] + head(data, -1)[,3]
	edata[seq(2,eenum,2),3] <- data[-1,2] - (head(data, -1)[,2] + head(data, -1)[,3])

	names(edata) <- c("event", "start", "duration")

	#filter out 0-duration operations
	fdata <- edata[edata[,3]>0,]

	return(list(fdata,
			c(0,1,2,3,4,5),
			c("host-to-device memcpy","device-to-host memcpy",
				"GPU kernel execution","diskIO","cudaMalloc","CPU Operation") ))
}

file2array3 = function(filename) {
	out <- read.table(filename, skip=10, colClasses="numeric")
	data <- out
	data[,3] = out[,3] - out[,2]

	enum <- dim(data)[1]
	ecol <- 1:3
	eenum<- 2*enum -1
	edata <- data 
	edata[eenum,3] <- 0

	#add "event 4" records
	# event 4, end, duration: current start -last end
	edata[seq(1,eenum,2),] <- data
	edata[seq(2,eenum,2),1] <- 4
	edata[seq(2,eenum,2),2] <- head(data, -1)[,2] + head(data, -1)[,3]
	edata[seq(2,eenum,2),3] <- data[-1,2] - (head(data, -1)[,2] + head(data, -1)[,3])

	avgE4d <- mean(edata[seq(2,eenum,2),3])
	edata[seq(2,eenum,2),3] <- avgE4d

	names(edata) <- c("event", "start", "duration")

	#filter out 0-duration operations
	fdata <- edata[edata[,3]>0,]

	return(list(fdata,
			c(0,1,2,3,4),
			c("host-to-device memcpy","device-to-host memcpy",
				"GPU kernel execution","CPU Operation","GPU malloc") ))
}

files2dur = function(files, reader=file2array3) {
    ret <- reader(files[1])
    etypes_id <- ret[[2]]
    etypes <- ret[[3]]

    queries <- rep(1, length(files))
    fid <- 1
    for (file in files) {
        T <- reader(file)[[1]]
        d = sum(T[, "duration"])
        queries[fid] <- d
        fid <- fid + 1
    }

    return(queries)
}
