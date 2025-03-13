import os
import time
import datetime
import decimal
import math
import psutil

import tabulate
import pandas as pd
import logging
import platform
import re

from dateutil import parser as date_parser
from typing import Union

#######################################################################################################################
# Version, version history
#######################################################################################################################
G_SourceVersion = 'SourceVersion : 20240726(NSCM)v2'
G_SourceVersion_hist = '''
-------------------
VERSION     HISTORY
-------------------
20240612    CREATED
20240613    add function : gfn_add_week
20240618    add function : gfn_get_partial_week, gfn_get_timedelta
20240710    PKG명 변경 테스트 (common -> NSCMCommon)
20240726    LWP Memory사용량 모니터링을 위해 log항목 추가
20240726    warn,error,critical log 사용을 위한 주석해제
'''


# 전역변수 : datetime 변환용
G_STR_DATE_FORMAT = '%Y-%m-%d'  # date format
G_STR_WEEK_FORMAT = '%Y%W'  # Week format
G_STR_MONTH_FORMAT = '%Y%m'  # Month format


# 전역변수 : local 설정 변수
G_PROGRAM_NAME = None
G_IS_Local = None


# round 설정
G_CONTEXT = decimal.getcontext()
G_CONTEXT.rounding = decimal.ROUND_HALF_UP
G_DECIMAL = decimal.Decimal


tabulate.PRESERVE_WHITESPACE = True
tabulate_args = {
    'headers': 'keys',
    # 'tablefmt': 'grid',  # simple is the default format
    'disable_numparse': True,
    'showindex': True,
}


# class Singleton(type):
#     _instances = {}
#
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]
#
#
# class G_Logger(metaclass=Singleton):
class G_Logger:
    """
    \n 로거 객체 생성:
    \n logger = G_Logger(p_py_name='template_training.py')
    \n \\
    \n 시작 로그 출력:
    \n logger.Start()
    \n \\
    \n 스텝 종료 구분 (스텝번호, 설명)
    \n logger.Step(1, 'Step Description')  # <- Step 함수는 각 Step 이 끝나는 부분에서 호출
    \n logger.Step(2, '확정구간 생성 완료')  #    각 Step 이 성공적으로 끝났음을 확인
    \n logger.Step(3)
    \n \\
    \n 종료 로그 출력
    \n logger.Finish()
    \n \\
    \n 일반 문자열 출력 (로그 메시지, 로그 레벨)
    \n logger.Note('Debug 레벨 로그 예시', 10)
    \n logger.Note('Info 레벨 로그 예시', 20)
    \n \\
    \n 데이터프레임 출력 (변수명, 데이터프레임명, format, row_num, pd_query, selected_columns)
    \n logger.PrintDF(df_fcst, 'Forecast', 1)  # 열 간격이 공백으로 출력
    \n logger.PrintDF(df_fcst, 'Forecast', 2)  # 열 간격이 세미콜론으로 구분되어 출력
    \n logger.PrintDF(df_fcst, 'Forecast', 1, 50)  # 50 줄만 출력 (기본: 5줄 출력)
    \n \\
    \n 컬럼 필터링: ['Item.DSM', 'Week'] 컬럼만 출력
    \n logger.PrintDF(df_fcst, 'Forecast', 1, 10, selected_columns=['Item.DSM', 'Week'])
    \n \\
    \n 쿼리 + 컬럼 필터링: 'Week' 컬럼에서 값이 12 인 데이터를 쿼리하고, 'Week' 컬럼만 출력
    \n logger.PrintDF(df_fcst, 'Forecast', 1, 10, pd_query='Week in (12,)', selected_columns=['Week'])
    \n \\
    \n 쿼리: 'Item.Item' 컬럼에서 값이 'Item.A1' 이거나 'Item.B2' 이면서,
    \n 'Week' 컬럼에서 값이 12 또는 27 또는 29 인 데이터 출력
    \n (컬럼 네임에 공백, 마침표 등이 있을 경우 백틱(`) 으로 감싼다.)
    \n logger.PrintDF(df_fcst, 'Forecast', 1, 10, pd_query='`Item.Item` in ("Item.A1", "Item.B2") & Week in (12, 27, 29)')
    """
    def __init__(self, p_py_name):
        self._logger = logging.getLogger('o9_logger')
        self._ts_start = time.time()  # TimeStamp (for Start_End)
        self._ts_step = time.time()  # TimeStamp (for Step)
        self.py_name = p_py_name  # python name
        self.prefix = '[NSCMLOG]'

        global G_PROGRAM_NAME, G_IS_Local
        G_PROGRAM_NAME = p_py_name
        G_IS_Local = gfn_get_isLocal()

    def get_level(self) -> int:
        return self._logger.level

    def _set_time_stamp(self, for_step: bool = False) -> None:
        if for_step:
            self._ts_step = time.time()
        else:
            self._ts_start = time.time()

    def _get_time_stamp(self, for_step: bool = False) -> time:
        if for_step:
            return self._ts_step
        else:
            return self._ts_start

    def _process_duration_time(self, for_step: bool = False) -> str:
        """
        - time diff 계산, (for_step 값에 따라 분기)
        - ts_step 리셋 (for step 인 경우에만,)
        :return: str
        """
        time_diff = time.time() - self._get_time_stamp(for_step)
        if for_step:
            self._set_time_stamp(for_step)

        return f'( {time_diff:.3f} sec)'

    # 데이터프레임 포매팅 (tabulate 라이브러리 이용하여 문자열로 변환)
    def _df_formatter(self,
                      df_in: pd.DataFrame,
                      print_format: int,
                      row_num: int,
                      pd_query: str = None,
                      selected_columns: list = None) -> str:
        """
        - 컬럼 네임에 dtypes 추가
        - 데이터프레임 Shape 출력 ( oo Rows x oo Columns )
        :param df_in:
        :param print_format: 1=tabulate, 2=csv
        :param row_num: 출력 행 [0:row_num]
        :param selected_columns:
        :param pd_query:
        :return: 문자열 ( 판다스 데이터프레임 -> str 로 변환 )
        """

        # copy_df 생성 (query 수행, row_num 제한, 컬럼 필터링)
        (copy_df, shape_query_result) = self._get_copy_df(
                                          df_in=df_in,
                                          row_num=row_num,
                                          pd_query=pd_query,
                                          selected_columns=selected_columns)

        # 위에서 리턴되는 copy_df 는 pd.DataFrame 이거나 '' 일 수 있다.
        if isinstance(copy_df, pd.DataFrame):
            # copy_df 데이터프레임을 (dtypes 추가) 문자열로 변환
            msg = self._convert_df_to_str(copy_df, print_format)

            # 데이터프레임 shape 문자열 생성 ( oo Rows x oo Columns )
            shape_df_in = df_in.shape
            shape_str = self._gen_shape_str(shape_df_in, shape_query_result)
        else:
            msg = ''
            shape_str = ''

        # 출력 문자열 리턴
        return f'{msg}{shape_str}'

    # 입력받은 데이터프레임의 복사본을 생성
    @staticmethod
    def _get_copy_df(
          df_in: pd.DataFrame,
          row_num: int,
          pd_query: str,
          selected_columns: list) -> tuple:
        """
        1. copy_df 생성
            1.1.1 pd_query 가 있으면 df_in 에 대해 DataFrame.query -> copy_df 생성
                  + Shape 정보 생성
                  + row_num 적용
            1.1.1 pd_query 가 없으면, df_in 의 데이터프레임에서 row_num 까지 복사 -> copy_df 생성
        2. selected_column 파라미터가 있다면 컬럼 필터링 추가 적용
        :param df_in:
        :param row_num:
        :param pd_query:
        :param selected_columns:
        :return:
        """

        # place holder
        shape_query_result = None

        # 1. copy_df 생성
        if isinstance(pd_query, str) and len(pd_query):
            copy_df = df_in.query(expr=pd_query)  # (, inplace=False): default
            shape_query_result = copy_df.shape  # query result 를 row_num 만큼만 보여주므로 shape 에 대한 정보를 추가 제공
            copy_df = copy_df[:row_num].copy()
        else:
            copy_df = df_in[:row_num].copy()

        # 2. selected_columns 파라미터가 있으면 해당 컬럼만 출력
        if isinstance(selected_columns, list) and selected_columns:
            copy_df = copy_df.get(selected_columns, default='')

        return copy_df, shape_query_result

    # copy_df 데이터프레임을 (dtypes 추가) 문자열로 변환 (tabulate 또는 pd.to_csv() 사용)
    def _convert_df_to_str(self, copy_df: pd.DataFrame, print_format: int) -> str:
        """
        tabulate 출력이면,
          1. DataFrame 의 컬럼 네임 + dtypes 추가
          2. 문자열로 변환

        csv 출력이면,
          1. 문자열로 변환
          2. dtypes 문자열 추가
        """
        msg = ''  # place holder
        if print_format == 1:
            # 컬럼 네임에 dtypes 추가
            copy_df = self._tabulate_column_w_dtypes(copy_df)
            # tabulate 라이브러리로 출력 (+개행)
            msg = f'{tabulate.tabulate(copy_df, **tabulate_args)}\n'
        elif print_format == 2:
            # 판다스의 .to_csv() 를 이용해서 출력
            msg = copy_df.to_csv(
                    sep=';',  # 구분자 (default ',')
                    line_terminator='\n')  # 개행문자 (optional, \n for linux, \r\n for Windows, i.e.)
            # 위의 to_csv() 에서 출력된 문자열을 dtypes 가 추가된 문자열로 바꾼다.
            msg = self._add_dtypes_to_csv_header(copy_df, msg)
        return msg

    # "DataFrame Shape: ..." 문자열 생성
    def _gen_shape_str(self, shape_df_in: tuple, shape_query_result) -> str:
        if shape_query_result is not None:
            df_query_result_shape_str = self._gen_plural_str(shape_query_result)
            shape_str = f'Query Result Shape: {df_query_result_shape_str}'
        else:
            df_in_shape_str = self._gen_plural_str(shape_df_in)
            shape_str = f'DataFrame Shape: {df_in_shape_str}'

        return shape_str

    # 'oo Rows x oo Columns' 문자열 생성 (복수형 표현, Plural)
    @staticmethod
    def _gen_plural_str(df_shape: tuple) -> str:
        (row_no, col_no) = df_shape
        row_str = f'{row_no} Row{"s"[:row_no ^ 1]}'  # row_no 가 1이 아니면 s 를 붙여서 복수형으로 표현
        col_str = f'{col_no} Column{"s"[:col_no ^ 1]}'
        shape_str = f'{row_str} x {col_str}'

        return shape_str

    # DataFrame 의 컬럼 네임 변경 (+ dtypes)
    @staticmethod
    def _tabulate_column_w_dtypes(copy_df: pd.DataFrame) -> pd.DataFrame:
        """
        입력받은 데이터프레임의 컬럼 네임에 dtypes 를 붙여서 리턴
        :param copy_df: 입력된 데이터프레임 (원본에 영향이 없도록 복사본이 생성되어 입력됨)
        :return: 컬럼 네임에 dtypes 가 추가되어 컬럼 네임이 변경된 데이터프레임
        """
        col_names = copy_df.columns.values
        if len(col_names):
            dtypes = [_.name for _ in copy_df.dtypes.tolist()]  # dtypes 를 가져와서 list 로 변환
            column_w_dtypes = [f'{_[0]}\n({_[1]})' for _ in zip(col_names, dtypes)]  # 결합 (column name + '\n' + dtype)
            new_col_names = dict(zip(col_names, column_w_dtypes))
            copy_df = copy_df.rename(new_col_names, axis='columns', inplace=False)  # 데이터프레임에서 컬럼 네임만 교체
        return copy_df

    # csv 출력 형식일 때 dtypes 문자열 추가
    @staticmethod
    def _add_dtypes_to_csv_header(copy_df: pd.DataFrame, to_csv_msg: str) -> str:
        """
        pd.to_csv() 의 결과 문자열에 dtypes 추가
        :param copy_df: dtypes 를 가져올 데이터프레임
        :param to_csv_msg: 데이터프레임을 출력한 문자열 (세미콜론 구분자)
        :return: 입력으로 받은 to_csv_msg 의 첫 번째 개행문자 다음에 dtypes 를 추가한 결과물
        """
        # 첫 번째 개행문자를 기준으로 헤더와 그 나머지 부분을 분리
        (head, tail) = ('', '')  # place holder
        if len(to_csv_msg) and ('\n' in to_csv_msg):
            (head, tail) = to_csv_msg.split('\n', maxsplit=1)

        # dtypes 를 세미콜론으로 이어붙이고 리턴 msg 를 만든다
        len_col_names = len(copy_df.columns.values)
        if len_col_names and len(head):
            dtypes = [f'({d_.name})' for d_ in copy_df.dtypes.tolist()]  # dtypes 를 가져와서
            dtypes = ';'.join(dtypes)  # 세미콜론으로 이어붙인다
            if head.count(';') == len_col_names:  # 구분자인 세미콜론의 개수는 컬럼명 개수보다 하나 적어야 하지만 만약 같다면,
                dtypes = f';{dtypes}'  # 데이터프레임 Index Name 의 빈자리라고 보고 세미콜론을 맨 앞에 하나 추가
            # dtypes 추가하여 msg 다시 생성
            to_csv_msg = f'{head}\n{dtypes}\n{tail}'

        return to_csv_msg

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    #   20240726    jaemin.im   warn,error,critical log 사용을 위한 주석해제
    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def Start(self):
        """
        \n 목적: 시작
        \n 활용법: 시작시 호출
        """
        # 인포 레벨로 Start 로그를 생성,
        self._emit_log(
            log_msg='Start',
            log_level=20)
        # Start 타임스탬프(self._ts_start)를 셋팅
        self._set_time_stamp(for_step=False)

    def Step(self, p_step_no: int = 0, p_step_desc: str = ''):
        """
        \n 목적: 단계별
        \n 활용법: Step 뒤에 호출

        \n - Parameter:
        \n p_step_no (*)
        \n p_step_desc
        """
        # 인포 레벨로 Step.00x; 로그를 생성
        self._emit_log(
            log_msg=f'Step.{int(float(p_step_no)):03};{p_step_desc}',
            log_level=20,
            with_duration=True,  # ( 0.00x sec) 문자열 생성
            for_step=True)       # Step 타임스탬프(self._ts_step)를 셋팅

    def Note(self, p_note: str = '', p_log_level: int = 10):
        """
        \n 목적: String 출력
        \n 활용법: 필요한 구문 호출 || 원하는 경우에만

        \n - Parameter:
        \n p_note (*)
        \n p_log_level: debug
        """
        self._emit_log(
            log_msg=p_note,
            log_level=p_log_level)

    def PrintDF(self,
                p_df: pd.DataFrame,
                p_df_name: str = '',
                p_log_level: int = 10,
                p_format: int = 1,
                p_row_num: int = 5,
                p_query: str = None,
                p_columns: list = None):

        """
        \n 목적: DF 출력
        \n 활용법: dataframe 출력

        \n - Parameter:
        \n  p_log_level: 10=DEBUG | 20=INFO
        \n  p_format: 1=Tabulate | 2=csv
        """

        formatted_df = self._df_formatter(
                           df_in=p_df,
                           print_format=p_format,
                           row_num=p_row_num,
                           pd_query=p_query,
                           selected_columns=p_columns)

        self._emit_log(
            log_msg=f'{p_df_name}\n{formatted_df}',
            log_level=p_log_level)

    def Error(self):
        """
        \n 목적: 에러
        """

        self._emit_log(
            log_msg='Error',
            log_level=20)

    def Finish(self, p_df=None):
        """
        \n 목적: 종료
        \n 활용법: 종료시 호출
        """

        self._emit_log(
            log_msg='Finish',
            log_level=20,
            with_duration=True,  # ( 0.00x sec) 문자열 생성
            for_step=False)      # Start ~ Finish 시간을 계산

        # local file 출력
        if G_IS_Local is True and p_df is not None:
            gfn_get_down_csv_file(p_dataframe=p_df, p_dir='output')

    def _emit_log(self,
                  log_msg: str,
                  log_level: int = 10,
                  with_duration: bool = False,
                  for_step: bool = False):



        #log_msg = f'{self.py_name};{log_msg}'
        #20240726   jaemin.im   LWP Memory사용량 모니터링을 위해 log항목 추가
        mem = memory_usage('info')
        log_msg = f'{self.py_name};{log_msg};{mem}'

        if with_duration:
            dur_time = self._process_duration_time(for_step)
            log_msg = f'{log_msg};{dur_time}'

        if log_level == 20:
            log_msg = f'{self.prefix};{log_msg}'
            self._logger.info(log_msg)
        elif log_level == 30:
            log_msg = f'{self.prefix};{log_msg}'
            self._logger.warning(log_msg)
        else:
            self._logger.debug(log_msg)

#20240726   jaemin.im   LWP Memory사용량 모니터링을 위해 log항목 추가
def memory_usage(message: str = 'debug'):
    #import psutil
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20  # byte to MB

    #msg = f'{message} memory usage {rss: 10.5f} MB'
    msg = f'{round(rss)} MB'
    #return msg, rss
    return msg

###########################################################
# G_log_level Class
###########################################################
class G_log_level:
    """
    G_log_level.debug
    """
    @staticmethod
    def debug(): return logging.DEBUG

    @staticmethod
    def info(): return logging.INFO

    @staticmethod
    def warning(): return logging.WARNING

    @staticmethod
    def error(): return logging.ERROR

    @staticmethod
    def critital(): return logging.CRITICAL


###########################################################
# Local 개발 환경 설정 함수 3개
#       gfn_get_isLocal
#       gfn_set_local_logfile
#       gfn_get_down_csv_file
###########################################################
def gfn_get_isLocal() -> bool:
    """
    Local 구분

    :return: bool
    """
    bool_local = False
    if platform.system() not in ['Linux']:
        bool_local = True
    return bool_local


def gfn_set_local_logfile() -> None:
    """
    Local log file 설정

    :return:
    """
    if G_IS_Local is True:
        logging.getLogger().setLevel(logging.DEBUG)

    if G_IS_Local is True and G_PROGRAM_NAME is not None:
        if os.path.exists('log'):
            log_file_name = G_PROGRAM_NAME.replace('py', 'log')
            log_file_name = f'log/{log_file_name}'
            if os.path.exists(log_file_name):
                os.remove(log_file_name)
            file_handler = logging.FileHandler(log_file_name, encoding='UTF-8')
            logging.getLogger('o9_logger').addHandler(file_handler)


def gfn_get_down_csv_file(p_dataframe: pd.DataFrame, p_dir='output') -> None:
    """
    Local에서 최종 Out Dataframe -> .csv 파일로 출력

    :param p_dataframe:
    :param p_dir:
    :return:
    """
    if G_IS_Local is True:
        if os.path.exists(p_dir):
            csv_date = datetime.datetime.now().strftime('%Y%m%d_%H_%M')
            csv_out_filename = f'{p_dir}/{csv_date}_out_{G_PROGRAM_NAME.replace(".py", "")}.csv'
            p_dataframe.to_csv(csv_out_filename, index=False)


###########################################################
# datetime 변환 함수 10개 예시
#          gfn_add_week         : gfn_add_week('202352', 2) -> '202402'
#          gfn_TimeDimToDate_W  : gfn_TimeDimToDate_W('2024W01') -> datetime(2024-01-01 00:00:00)
#          gfn_TimeDateToChar_W : gfn_TimeDateToChar_W(2024.01.01 03:09:09) -> '2024W01'
#          gfn_TimeDimToDate_M  : gfn_TimeDimToDate_M('2024M01') -> datetime(2024-01-01 00:00:00)
#          gfn_TimeDateToChar_M : gfn_TimeDateToChar_M(2024.01.01 03:09:09) -> '2024M01'
#          gfn_to_date          : gfn_to_date('2024-W01', '%Y-W%W') -> datetime(2024-01-01 00:00:00)
#          gfn_to_char          : gfn_to_char(datetime(2024-01-01 00:00:00), '%Y-W%W') -> 2024-W01
#          gfn_is_date_parsing
#          gfn_is_date_matching
#          gfn_get_df_mst_week           : df_mst_week2 = gfn_get_df_mst_week(p_frist_week='202401', p_duration_week=30, p_in_out_week_format='%Y%W')
#          gfn_get_df_mst_week_from_date : df_mst_week5 = gfn_get_df_mst_week_from_date(p_frist_day='2024-01-01', p_duration_week=30, p_in_out_week_format='%Y%W', p_in_out_day_format='%Y-%m-%d')
#          gfn_get_partial_week : gfn_get_partial_week(p_datetime=datetime.datetime.now(), p_bool_FI_week=True) -> '202424A'
#          gfn_get_timedelta    : gfn_get_timedelta(p_float_day=5.5) -> datetime.timedelta(days=5, seconds=43200)
###########################################################
def gfn_add_week(p_str_yyyyww: str, p_week_delta=0) -> str:
    """
    string -> string
    ex) gfn_add_week('202352', 2) -> '202402'

    :param p_str_yyyyww:
    :param p_week_delta:
    :return:
    """
    result = None
    str_msg = ''
    if r'%W' in G_STR_WEEK_FORMAT and gfn_is_date_matching(p_str_yyyyww, G_STR_WEEK_FORMAT):
        year, week = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_yyyyww)
        if len(all_char) == 6:
            year = int(all_char[:4])
            week = int(all_char[4:])
        elif len(all_char) == 5:
            year = int(all_char[:4])
            week = int(all_char[-1:])
        else:
            str_msg = f'''Error : week format string not matching
            common function : gfn_add_week -> gfn_is_date_matching
            param    : ({p_str_yyyyww}, {str(p_week_delta)})
            '''
            raise Exception(str_msg)
        # return datetime.datetime.fromisocalendar(year, week, p_week_day)
        result = datetime.datetime.strptime(f"{year:04d}{week:02d}{1:d}", "%G%V%u")  # .date()

    else:
        str_msg = f'''Error : week format string not matching
        common function : gfn_add_week -> gfn_is_date_matching
        param    : ({p_str_yyyyww}, {p_week_delta})
        format   : {G_STR_WEEK_FORMAT}
        '''
        raise Exception(str_msg)

    p_day_delta = p_week_delta * 7
    return gfn_to_char(p_datetime=result, p_format=G_STR_WEEK_FORMAT, p_day_delta=p_day_delta)


def gfn_TimeDimToDate_W(p_str_datetype: str, p_week_day=1) -> datetime:
    """
    string -> datetime
    ex) gfn_TimeDimToDate_W('2024W01') -> datetime(2024-01-01 00:00:00)

    :param p_str_datetype:
    :param p_week_day:
    :return:
    """
    str_msg = ''
    if r'%W' in G_STR_WEEK_FORMAT and gfn_is_date_matching(p_str_datetype, G_STR_WEEK_FORMAT):
        year, week = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = int(all_char[:4])
            week = int(all_char[4:])
        elif len(all_char) == 5:
            year = int(all_char[:4])
            week = int(all_char[-1:])
        else:
            str_msg = f'''Error : week format string not matching
            common function : gfn_TimeDimToDate_W -> gfn_is_date_matching
            param    : ({p_str_datetype}, {p_week_day})
            '''
            raise Exception(str_msg)
        # return datetime.datetime.fromisocalendar(year, week, p_week_day)
        return datetime.datetime.strptime(f"{year:04d}{week:02d}{p_week_day:d}", "%G%V%u")  # .date()
    else:
        str_msg = f'''Error : week format string not matching
        common function : gfn_TimeDimToDate_W -> gfn_is_date_matching
        param    : ({p_str_datetype}, {p_week_day})
        format   : {G_STR_WEEK_FORMAT}
        '''
        raise Exception(str_msg)


def gfn_TimeDateToChar_W(p_datetime: datetime) -> str:
    """
    datetime -> string
    ex) gfn_TimeDateToChar_W(2024.01.01 03:09:09) -> '2024W01'

    :param p_datetime:
    :return:
    """
    str_msg = ''
    if r'%W' in G_STR_WEEK_FORMAT:
        # year = str(p_datetime.isocalendar().year)
        # week = str(p_datetime.isocalendar().week).zfill(2)
        year = str(p_datetime.isocalendar()[0])
        week = str(p_datetime.isocalendar()[1]).zfill(2)
        return G_STR_WEEK_FORMAT.replace('%Y', year).replace('%W', week)
    else:
        str_msg = f'''Error : format string not matching
        common function : gfn_TimeDateToChar_W -> gfn_is_date_matching
        param    : ({p_datetime})
        format   : {G_STR_WEEK_FORMAT}
        '''
        raise Exception(str_msg)


def gfn_TimeDimToDate_M(p_str_datetype: str) -> datetime:
    """
    string -> datetime
    ex) gfn_TimeDimToDate_M('2024M01') -> datetime(2024-01-01 00:00:00)

    :param p_str_datetype:
    :return:
    """
    str_msg = ''
    if r'%m' in G_STR_MONTH_FORMAT and gfn_is_date_matching(p_str_datetype, G_STR_MONTH_FORMAT):
        year, month = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = all_char[:4]
            month = all_char[4:]
        elif len(all_char) == 5:
            year = all_char[:4]
            month = all_char[-1:]
        else:
            str_msg = f'''Error : month format string not matching
            common function : gfn_TimeDimToDate_M -> gfn_is_date_matching
            param    : ({p_str_datetype})
            '''
            raise Exception(str_msg)
        str_datetype = '-'.join([year, month, '01'])
        return datetime.datetime.strptime(str_datetype, '%Y-%m-%d')
    else:
        str_msg = f'''Error : month format string not matching
        common function : gfn_TimeDimToDate_M -> gfn_is_date_matching
        param    : ({p_str_datetype})
        format   : {G_STR_MONTH_FORMAT}
        '''
        raise Exception(str_msg)


def gfn_TimeDateToChar_M(p_datetime: datetime) -> str:
    """
    datetime -> string
    ex) gfn_TimeDateToChar_M(2024.01.01 03:09:09) -> '2024M01'

    :param p_datetime:
    :return:
    """
    str_msg = ''
    if r'%m' in G_STR_MONTH_FORMAT:
        year = p_datetime.strftime('%Y')
        month = p_datetime.strftime('%m')
        return G_STR_MONTH_FORMAT.replace('%Y', year).replace('%m', month)
    else:
        str_msg = f'''Error : month format string not matching
        common function : gfn_TimeDateToChar_M -> gfn_is_date_matching
        param    : ({p_datetime})
        format   : {G_STR_MONTH_FORMAT}
        '''
        raise Exception(str_msg)


def gfn_to_date(p_str_datetype: str, p_format: str, p_week_day=1, p_day_delta=0) -> datetime:
    """
    string -> datetime
    ex) gfn_to_date('2024-W01', '%Y-W%W') -> datetime(2024-01-01 00:00:00)
        gfn_to_date('2024-M01', '%Y-M%m') -> datetime(2024-01-01 00:00:00)
        gfn_to_date('20240101', '%Y%m%d') -> datetime(2024-01-01 00:00:00)
        gfn_to_date('2024.01.01', '%Y.%m.%d') -> datetime(2024-01-01 00:00:00)
        gfn_to_date('2024.01.01 03:09:09', '%Y.%m.%d %H:%M:%S') -> datetime(2024-01-01 03:09:09)

    :param p_str_datetype:
    :param p_format:
    :param p_week_day:
    :param p_day_delta:
    :return:
    """
    result = None
    str_msg = ''
    if r'%W' in p_format and gfn_is_date_matching(p_str_datetype, p_format):
        year, week = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = int(all_char[:4])
            week = int(all_char[4:])
        elif len(all_char) == 5:
            year = int(all_char[:4])
            week = int(all_char[-1:])
        else:
            str_msg = f'''Error : week format string not matching
            common function : gfn_to_date -> gfn_is_date_matching
            param    : ({p_str_datetype}, {p_format}, {p_week_day})
            '''
            raise Exception(str_msg)

        # result = datetime.datetime.fromisocalendar(year, week, p_week_day)
        result = datetime.datetime.strptime(f"{year:04d}{week:02d}{p_week_day:d}", "%G%V%u")  # .date()

    elif r'%m' in p_format and r'%d' not in p_format and gfn_is_date_matching(p_str_datetype, p_format):
        year, month = None, None
        all_char = re.sub(r'[^0-9]', '', p_str_datetype)
        if len(all_char) == 6:
            year = all_char[:4]
            month = all_char[4:]
        elif len(all_char) == 5:
            year = all_char[:4]
            month = all_char[-1:]
        else:
            str_msg = f'''Error : month format string not matching
            common function : gfn_to_date -> gfn_is_date_matching
            param    : ({p_str_datetype})
            '''
            raise Exception(str_msg)
        str_datetype = '-'.join([year, month, '01'])

        result = datetime.datetime.strptime(str_datetype, '%Y-%m-%d')
    else:
        if gfn_is_date_parsing(p_str_datetype):
            if gfn_is_date_matching(p_date_str=p_str_datetype, p_format=p_format):
                result = datetime.datetime.strptime(p_str_datetype, p_format)
            else:
                str_msg = f'''Error : format string not matching
                common function : gfn_to_date -> gfn_is_date_matching
                param    : ({p_str_datetype}, {p_format}, {p_week_day})
                '''
                raise Exception(str_msg)
        else:
            str_msg = f'''Error : format string not parsing
            common function : gfn_to_date -> gfn_is_date_parsing
            param    : ({p_str_datetype}, {p_format}, {p_week_day})
            '''
            raise Exception(str_msg)

    if p_day_delta == 0:
        return result
    else:
        return result + datetime.timedelta(days=p_day_delta)


def gfn_to_char(p_datetime: datetime, p_format: str, p_day_delta=0) -> str:
    """
    string -> datetime
    ex) gfn_to_char(datetime(2024-01-01 00:00:00), '%Y-W%W') -> 2024-W01
        gfn_to_char(datetime(2024-01-01 00:00:00), '%Y-M%m') -> 2024-M01
        gfn_to_char(datetime(2024-01-01 00:00:00), '%Y%m%d') -> 20240101
        gfn_to_date(datetime(2024-01-01 00:00:00), '%Y.%m.%d') -> 2024.01.01
        gfn_to_date(datetime(2024-01-01 03:09:09), '%Y.%m.%d %H:%M:%S') -> 2024.01.01 03:09:09

    :param p_datetime:
    :param p_format:
    :param p_day_delta:
    :return:
    """
    result = None
    str_msg = ''
    if p_day_delta != 0:
        p_datetime = p_datetime + datetime.timedelta(days=p_day_delta)

    if r'%W' in p_format:
        # year = str(p_datetime.isocalendar().year)
        # week = str(p_datetime.isocalendar().week).zfill(2)
        year = str(p_datetime.isocalendar()[0])
        week = str(p_datetime.isocalendar()[1]).zfill(2)
        result = p_format.replace('%Y', year).replace('%W', week)

    elif r'%m' in p_format and r'%d' not in p_format:
        year = p_datetime.strftime('%Y')
        month = p_datetime.strftime('%m')
        result = p_format.replace('%Y', year).replace('%m', month)

    else:
        if gfn_is_date_matching(p_date_str=p_datetime, p_format=p_format):
            result = datetime.datetime.strftime(p_datetime, p_format)
        else:
            str_msg = f'''Error : format string not matching
            common function : gfn_to_char -> gfn_is_date_matching
            param    : ({p_datetime}, {p_format}, {p_day_delta})
            '''
            raise Exception(str_msg)

    return result


def gfn_is_date_parsing(p_date_str: str) -> bool:
    try:
        return bool(date_parser.parse(p_date_str))
    except ValueError:
        return False


def gfn_is_date_matching(p_date_str: Union[str, datetime.datetime], p_format) -> bool:
    try:
        if isinstance(p_date_str, str):
            return bool(datetime.datetime.strptime(p_date_str, p_format))
        else:
            return bool(datetime.datetime.strftime(p_date_str, p_format))
    except ValueError:
        return False


def gfn_get_df_mst_week(p_frist_week: str, p_duration_week=None, p_in_out_week_format=G_STR_WEEK_FORMAT, p_duration_day=None) -> pd.DataFrame:
    """
    week 기준 dataframe 생성
    df_mst_week = gfn_get_df_mst_week(p_frist_week='202401', p_duration_week=30)
    df_mst_week = gfn_get_df_mst_week(p_frist_week='202401', p_duration_week=30, p_in_out_week_format='%Y%W')
    df_mst_week = gfn_get_df_mst_week(p_frist_week='202401', p_duration_day=365)

    :param p_frist_week:
    :param p_duration_week:
    :param p_in_out_week_format:
    :param p_duration_day:
    :return:
    """
    date_start = gfn_to_date(p_str_datetype=p_frist_week, p_format=p_in_out_week_format)

    list_loop = []
    if p_duration_day is None:
        for i in range(0, p_duration_week):
            list_loop.append(i*7)
    else:
        for i in range(0, p_duration_day, 7):
            list_loop.append(i)

    list_result_week = []
    for i, value in enumerate(list_loop):
        str_week = gfn_to_char(p_datetime=date_start, p_format=p_in_out_week_format, p_day_delta=value)
        list_result_week.append([str_week, 1])
    return pd.DataFrame(list_result_week, columns=['week', 'key'])


def gfn_get_df_mst_week_from_date(p_frist_day: Union[str, datetime.datetime], p_duration_week=None,
                              p_in_out_week_format=G_STR_WEEK_FORMAT, p_in_out_day_format=G_STR_DATE_FORMAT, p_duration_day=None) -> pd.DataFrame:
    """
    week 기준 dataframe 생성
    df_mst_week = gfn_get_df_mst_week_from_date(p_frist_day='20240101', p_duration_week=30)
    df_mst_week = gfn_get_df_mst_week_from_date(p_frist_day='2024-01-01', p_duration_week=30, p_in_out_week_format='%Y%W', p_in_out_day_format='%Y-%m-%d')
    df_mst_week = gfn_get_df_mst_week_from_date(p_frist_day=datetime.datetime.now(), p_duration_day=365)

    :param p_frist_day:
    :param p_duration_week:
    :param p_in_out_week_format:
    :param p_in_out_day_format:
    :param p_duration_day:
    :return:
    """
    if isinstance(p_frist_day, str):
        date_start = gfn_to_date(p_str_datetype=p_frist_day, p_format=p_in_out_day_format)
    else:
        date_start = p_frist_day

    list_loop = []
    if p_duration_day is None:
        for i in range(0, p_duration_week):
            list_loop.append(i * 7)
    else:
        for i in range(0, p_duration_day, 7):
            list_loop.append(i)

    list_result_week = []
    for i, value in enumerate(list_loop):
        str_week = gfn_to_char(p_datetime=date_start, p_format=p_in_out_week_format, p_day_delta=value)
        list_result_week.append([str_week, 1])
    return pd.DataFrame(list_result_week, columns=['week', 'key'])


# parsial week
def gfn_get_partial_week(p_datetime: datetime, p_bool_FI_week: bool = False) -> str:
    """
    parsial week 반환
    ex) gfn_get_partial_week(p_datetime=datetime.datetime.now(), p_bool_FI_week=True) -> '202424A'
        gfn_get_partial_week(p_datetime=datetime.datetime.now())                      -> '202424'

    :param p_datetime:
    :param p_bool_FI_week:
    :return:
    """
    year = str(p_datetime.isocalendar()[0])
    week = str(p_datetime.isocalendar()[1]).zfill(2)
    month = p_datetime.strftime('%m')
    # iyyyiw
    iyyyiw = G_STR_WEEK_FORMAT.replace('%Y', year).replace('%W', week)

    # start date
    monday_mm = gfn_to_date(p_str_datetype=iyyyiw, p_format='%Y%W', p_week_day=1).strftime('%m')
    # end date
    sunday_mm = gfn_to_date(p_str_datetype=iyyyiw, p_format='%Y%W', p_week_day=7).strftime('%m')

    if (month == monday_mm) & (month == sunday_mm):
        if p_bool_FI_week:
            result = iyyyiw + 'A'
        else:
            result = iyyyiw
    elif (month == monday_mm) & (month != sunday_mm):
        result = iyyyiw + 'A'
    elif (month != monday_mm) & (month == sunday_mm):
        result = iyyyiw + 'B'
    else:
        result = iyyyiw
    return result


def gfn_get_timedelta(p_float_day: float) -> datetime.timedelta:
    """
    day값 -> timedelta 변환
    ex) gfn_get_timedelta(p_float_day=5.5) -> datetime.timedelta(days=5, seconds=43200)

    :param p_float_day:
    :return:
    """
    int_day = math.floor(p_float_day)
    float_day = p_float_day - int_day
    int_second = float_day * 24 * 3600
    return datetime.timedelta(days=int_day, seconds=int_second)


###########################################################
# round 함수
#       gfn_get_round
###########################################################
def gfn_get_round(p_float: Union[str, float], p_decimal=0) -> decimal.Decimal:
    """
    # round 예시
    print(gfn_get_round(2.999999999999975, p_decimal=8))
    df['col1 -> common apply'] = df['col1'].apply(gfn_get_round)
    df['col1 -> common map']   = df['col1'].map(gfn_get_round)
    df['col2 -> common round 2'] = df['col2'].apply(gfn_get_round, p_decimal=2)

    :param p_float:
    :param p_decimal:
    :return:
    """
    if isinstance(p_float, float):
        p_float = str(p_float)
    return round(G_DECIMAL(p_float), p_decimal)


###########################################################
# 최빈값 설정 함수
#       gfn_set_rep
###########################################################
def gfn_set_rep(p_df: pd.DataFrame, p_list_key: list, p_str_rep_column: str) -> tuple:
    """
    # 최빈값 설정 예시 : return tuple
    (최빈값을 반영한 dataframe, 최빈값 master dataframe)
    df_in_rep, df_mst_rep = gfn_set_rep(df_in_rep, p_list_key=['DC'], p_str_rep_column='REP_DC')

    :param p_df:
    :param p_list_key:
    :param p_str_rep_column:
    :return:
    """
    list_all = p_list_key + [p_str_rep_column]
    list_sort = p_list_key + ['set_rep.count']
    list_ascending = [True for i in list_sort if i not in ['set_rep.count']] + [False]

    _df_rep = p_df[list_all].copy()

    _df_rep['set_rep.count'] = 1
    _df_rep = _df_rep.groupby(list_all)['set_rep.count'].sum().reset_index()

    _df_rep = _df_rep.sort_values(by=list_sort, ascending=list_ascending).reset_index(drop=True)

    _df_rep['set_rep.rank'] = _df_rep.groupby(p_list_key)['set_rep.count'].rank(method='min', ascending=False)
    _df_rep = _df_rep.loc[_df_rep['set_rep.rank'] == 1]

    _df_rep.drop_duplicates(p_list_key, keep='first', inplace=True)
    _df_rep = _df_rep.drop(['set_rep.count', 'set_rep.rank'], axis=1)  # column 제거

    p_df = p_df.drop([p_str_rep_column], axis=1)  # column 제거
    return (pd.merge(p_df, _df_rep, how='left', on=p_list_key), _df_rep.copy())
