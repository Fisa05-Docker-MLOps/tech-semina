import logging
import logging.handlers

def setup_logger(name: str, level=logging.INFO):
    """
    표준 로거를 설정하고 반환합니다.
    - 콘솔 핸들러와 파일 핸들러를 모두 가집니다.
    - 파일 핸들러는 로그 파일을 매일 자정에 새로 생성합니다.
    """
    # 1. 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 2. 포매터 생성 (로그 형식 정의)
    # 예: 2025-08-18 17:30:00,123 - my_app - INFO - This is a message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 3. 핸들러 생성 (콘솔 및 파일)
    # 3-1. 콘솔 핸들러: 로그를 콘솔에 출력
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 콘솔에는 INFO 레벨 이상만 출력
    console_handler.setFormatter(formatter)

    # 3-2. 파일 핸들러: 로그를 파일에 저장
    # TimedRotatingFileHandler: 특정 시간 간격(자정)으로 로그 파일을 새로 생성
    # when='midnight': 자정마다, interval=1: 매일, backupCount=7: 최대 7개 파일 보관
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=f'{name}.log',
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # 파일에는 DEBUG 레벨 이상 모두 기록
    file_handler.setFormatter(formatter)

    # 4. 로거에 핸들러 추가
    # 핸들러가 이미 추가되어 있는지 확인하여 중복 추가 방지
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
