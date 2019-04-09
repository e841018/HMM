.PHONY: all log
.PHONY: clean clean_data

LDFLAGS+=-lm     # link to math library
TARGET=train test
iter=100

# accuracy may differ slightly between make all and make log
# due to passing double in string format

all: $(TARGET) clean_data
	# train
	for i in {1..5}; do \
		./train \
			${iter} \
			../model_init.txt \
			../seq_model_0$${i}.txt \
			model_0$${i}.txt \
	;done
	# test
	for i in {1..2}; do \
		./test \
			../modellist.txt \
			../testing_data$${i}.txt \
			result$${i}.txt \
	;done

log: $(TARGET) clean_data
	# copy models
	for i in {1..5}; do \
		cp \
			../model_init.txt \
			model_0$${i}.txt \
	;done
	# test before training
	./test \
		../modellist.txt \
		../testing_data1.txt \
		result1.txt \
		| sed "s/^/0 /" \
		>> log.txt
	for i in {1..5}; do \
		./test \
			../modellist.txt \
			../seq_model_0$${i}.txt \
			result1.txt \
			model_0$${i}.txt \
			| sed "s/^/$${it} /" \
			>> log$${i}.txt \
	;done
	# iterations
	for it in {1..${iter}}; do \
		for i in {1..5}; do \
			./train \
				1 \
				model_0$${i}.txt \
				../seq_model_0$${i}.txt \
				model_0$${i}.txt \
		;done ; \
		./test \
			../modellist.txt \
			../testing_data1.txt \
			result1.txt \
			| sed "s/^/$${it} /" \
			>> log.txt ; \
		for i in {1..5}; do \
			./test \
				../modellist.txt \
				../seq_model_0$${i}.txt \
				result1.txt \
				model_0$${i}.txt \
				| sed "s/^/$${it} /" \
				>> log$${i}.txt \
		;done \
	;done
	# generate results1~2.txt
	for i in {1..2}; do \
		./test \
			../modellist.txt \
			../testing_data$${i}.txt \
			result$${i}.txt \
	;done

clean:
	$(RM) $(TARGET)

clean_data:
	$(RM) model_0*.txt
	$(RM) result*.txt
	$(RM) acc.txt
	$(RM) log*.txt

