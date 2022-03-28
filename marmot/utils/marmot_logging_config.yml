version: 1
formatters:
  info_format:
    format: '%(asctime)s:%(levelname)-8s- %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'
  warning_format:
    format: '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s - %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    stream  : ext://sys.stdout

  warning_handler:
    class: logging.handlers.RotatingFileHandler
    level: WARNING
    formatter: warning_format
    filename: '{}/WARNINGS_{}{}.log'
    backupCount: 3
    encoding: utf8
    mode: a

  info_handler:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: info_format
    filename: '{}/Log_{}{}.log'
    backupCount: 3
    encoding: utf8
    mode: a

loggers:
  formatter:
    level: INFO
    handlers: [console,warning_handler,info_handler]
    propagate: yes

  plotter:
    level: INFO
    handlers: [console,warning_handler,info_handler]
    propagate: yes

  root:
    level: DEBUG
    handlers: [console]