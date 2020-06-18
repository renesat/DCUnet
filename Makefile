NAME=vkinternship
DOCKER_NAME=${NAME}
DOCKER_MAIN_DIR=/workdir
NEURO_DOCKER_NAME=image:${DOCKER_NAME}
NEURO=neuro
PRESET?=cpu-small

PROJECT_PATH=${NAME}
PROJECT_PATH_STORAGE=storage:${PROJECT_PATH}
NOTEBOOKS_PATH=notebooks
CODE_PATH=src
DATA_PATH=data
RESULTS_PATH=results
CONFIG_PATH=config.yml

make-storage:
	$(NEURO) mkdir --parents ${PROJECT_PATH_STORAGE} \
		${PROJECT_PATH_STORAGE}/${CODE_PATH} \
		${PROJECT_PATH_STORAGE}/${DATA_PATH} \
		${PROJECT_PATH_STORAGE}/${NOTEBOOKS_PATH} \
		${PROJECT_PATH_STORAGE}/${RESULTS_PATH} \
		${PROJECT_PATH_STORAGE}/${RESULTS_PATH}/tb
	$(NEURO) cp \
		--update \
		run_notebook.sh ${PROJECT_PATH_STORAGE}/
	$(NEURO) cp \
		--update \
		run_tensorboard.sh ${PROJECT_PATH_STORAGE}/

upload-all: upload-code upload-notebooks upload-data

upload-code:
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(CODE_PATH) ${PROJECT_PATH_STORAGE}/${CODE_PATH}

upload-data:
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		$(DATA_PATH) ${PROJECT_PATH_STORAGE}/${DATA_PATH}

download-data:
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		${PROJECT_PATH_STORAGE}/${DATA_PATH} $(DATA_PATH)


upload-notebooks:
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		--exclude="*" \
		--include="*.ipynb" \
		$(NOTEBOOKS_PATH) ${PROJECT_PATH_STORAGE}/${NOTEBOOKS_PATH}

download-notebooks:
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		--exclude="*" \
		--include="*.ipynb" \
		${PROJECT_PATH_STORAGE}/${NOTEBOOKS_PATH} $(NOTEBOOKS_PATH)


make-docker:
	docker build -t ${DOCKER_NAME} .

upload-docker:
	neuro image push ${DOCKER_NAME} "${NEURO_DOCKER_NAME}"

run-neuro-jupyter:
	neuro run -n jupyter -s ${PRESET} \
		--http 8888 \
		--http-auth \
		--browse \
		--detach \
		--volume ${PROJECT_PATH_STORAGE}:${DOCKER_MAIN_DIR}:rw \
		--life-span=1d \
		"${NEURO_DOCKER_NAME}" bash run_notebook.sh

run-neuro-tensorboard:
	neuro run -n tensorboard -s ${PRESET} \
		--http 6006 \
		--http-auth \
		--browse \
		--detach \
		--volume ${PROJECT_PATH_STORAGE}:${DOCKER_MAIN_DIR}:rw \
		--life-span=1d \
		"${NEURO_DOCKER_NAME}" bash run_tensorboard.sh
