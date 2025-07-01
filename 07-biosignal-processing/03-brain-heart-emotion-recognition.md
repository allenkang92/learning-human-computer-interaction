# 7.3 뇌와 심장 기반의 감성 인식 시스템

## 개요

뇌파(EEG)와 심전도(ECG) 신호는 감성 인식 시스템에서 가장 중요한 생체신호 중 하나입니다. 이 장에서는 오픈소스 기반의 뇌정보 처리 및 추출 방법과 심장 활동 분석을 통한 감성 인식 시스템에 대해 학습합니다.

## 1. 오픈소스 기반의 뇌정보 처리 및 추출

### 뇌파 개요 및 측정 원리

#### 뇌파(EEG)의 정의
- **뇌파**: 뇌신경에서 발생하는 전기적 신호가 합성되어 나타나는 신호
- **측정 방법**: 미세한 뇌 표면의 신호를 전극을 이용해 측정한 전위
- **특징**: 뇌의 다양한 활동 상태를 실시간으로 반영하는 비침습적 신호

#### 뇌 영역별 기능과 역할

**전두엽 (Frontal Lobe)**:
- 이성적 사고, 기억 및 계획 수립
- 의사 결정, 인격 형성 담당
- 감정 조절과 실행 기능

**측두엽 (Temporal Lobe)**:
- 감정 표현, 단기 기억
- 좌측 측두엽: 언어 해석 기능
- 청각 정보 처리

**두정엽 (Parietal Lobe)**:
- 시각, 촉각, 통각, 미각 등 감각 정보의 통합
- 공간 인식 및 주의 집중
- 감각 정보의 해석

**후두엽 (Occipital Lobe)**:
- 시각 정보 처리
- 시각적 인식과 해석
- 공간적 시각 처리

### 뇌파의 종류

#### 1. 자발 뇌파 (Spontaneous EEG)

**특징**:
- 감성적 변화(초조, 불안 등)에 따라 변화
- 인지적 처리(계산, 판단 등) 상태 반영
- 신체적 상태(졸림, 피곤 등) 표현
- 자연스럽게 발생하는 신호

**활용**:
- 현재의 심리적 상태 파악
- 인지적 처리 속도 측정
- 감성 상태 모니터링

#### 2. 유발 뇌파 (Event Related Potential, ERP)

**특징**:
- 외부적 자극을 통한 감각기 자극(시각, 청각, 촉각 등) 시 측정
- 외부적 자극에 동기화된 반응
- BCI(Brain Computer Interface) 분야에 광범위하게 사용

**활용**:
- 인지 과정 분석
- 뇌-컴퓨터 인터페이스 구현
- 신경 마케팅 연구

### EEG 측정 원리 및 전극 배치

#### 국제 10-20 시스템
- **뇌 영역별 이니셜 사용**:
  - F (Frontal): 전두엽
  - T (Temporal): 측두엽
  - P (Parietal): 두정엽
  - O (Occipital): 후두엽

#### 측정 방법 분류

**침습형 뇌파 측정 (Invasive EEG)**:
- 전극을 뇌조직에 직접 삽입
- 높은 공간 해상도와 신호 품질
- 의료용으로 제한적 사용

**비침습형 뇌파 측정 (Non-Invasive EEG)**:
- 두피 표면에 전극 부착
- 안전하고 사용하기 쉬움
- 일반적인 연구 및 응용에 사용

## 2. 뇌파 신호 처리

### 신호 처리 파이프라인

#### 1단계: 뇌파 신호 수집
```python
# 뇌파 데이터 수집 예시
import numpy as np
import scipy.signal as signal

# 샘플링 주파수 설정
fs = 1000  # 1000 Hz

# 뇌파 신호 수집 (예시)
def collect_eeg_data(duration=10):
    """
    지정된 시간 동안 뇌파 데이터 수집
    """
    samples = int(duration * fs)
    # 실제로는 하드웨어에서 데이터 수집
    # 여기서는 시뮬레이션 데이터 생성
    time = np.linspace(0, duration, samples)
    
    # 다양한 주파수 성분을 가진 신호 생성
    alpha = np.sin(2 * np.pi * 10 * time)  # 10Hz 알파파
    beta = np.sin(2 * np.pi * 20 * time)   # 20Hz 베타파
    noise = np.random.normal(0, 0.1, samples)  # 노이즈
    
    eeg_signal = alpha + 0.5 * beta + noise
    return time, eeg_signal
```

#### 2단계: 노이즈 처리
```python
def preprocess_eeg(eeg_signal, fs):
    """
    뇌파 신호 전처리
    - 안륜근 움직임, 눈 깜빡임 등의 아티팩트 제거
    - 하드웨어적 필터링을 통한 노이즈 제거
    """
    # 밴드패스 필터 (1-50Hz)
    nyquist = fs / 2
    low_freq = 1.0 / nyquist
    high_freq = 50.0 / nyquist
    
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    filtered_signal = signal.filtfilt(b, a, eeg_signal)
    
    # 노치 필터 (50Hz 전원선 노이즈 제거)
    notch_freq = 50.0 / nyquist
    b_notch, a_notch = signal.iirnotch(notch_freq, 30)
    cleaned_signal = signal.filtfilt(b_notch, a_notch, filtered_signal)
    
    return cleaned_signal
```

#### 3단계: 주파수 변환
```python
def frequency_analysis(eeg_signal, fs):
    """
    시간 영역 신호를 주파수 영역으로 변환
    """
    # FFT를 통한 주파수 분석
    frequencies = np.fft.fftfreq(len(eeg_signal), 1/fs)
    fft_values = np.fft.fft(eeg_signal)
    power_spectrum = np.abs(fft_values) ** 2
    
    # 양의 주파수만 사용
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    power_spectrum = power_spectrum[positive_freq_idx]
    
    return frequencies, power_spectrum
```

#### 4단계: 주파수 파워 값 추출
```python
def extract_frequency_bands(frequencies, power_spectrum):
    """
    뇌파 주파수 대역별 파워 추출
    """
    bands = {
        'delta': (1, 4),    # 델타파: 1-4Hz
        'theta': (4, 8),    # 세타파: 4-8Hz
        'alpha': (8, 15),   # 알파파: 8-15Hz
        'beta': (16, 30),   # 베타파: 16-30Hz
        'gamma': (32, 50)   # 감마파: 32-50Hz
    }
    
    band_powers = {}
    
    for band_name, (low_freq, high_freq) in bands.items():
        # 해당 주파수 대역의 인덱스 찾기
        band_idx = (frequencies >= low_freq) & (frequencies <= high_freq)
        
        # 해당 대역의 평균 파워 계산
        band_power = np.mean(power_spectrum[band_idx])
        band_powers[band_name] = band_power
    
    return band_powers
```

### 뇌파 주파수 대역별 특성

#### 델타파 (Delta, 1-4Hz)
- **특성**: 수면파, 수면 시 크게 활성화
- **생리적 의미**: 깊은 수면 상태, 무의식 상태
- **감성 인식**: 의식 수준, 각성도 측정

#### 세타파 (Theta, 4-7Hz)
- **특성**: 정서적 안정, 기억과 관련된 일 수행 시 활성화
- **생리적 의미**: 깊은 이완, 명상 상태, 창의적 사고
- **감성 인식**: 스트레스 감소, 이완 상태 지표

#### 알파파 (Alpha, 8-15Hz)
- **특징**: 편안하고 안정 상태일수록 활성화
- **시각적 차단**: 시각적 정보가 차단되면 개안 대비 알파 활성화
- **생리적 의미**: 이완된 각성 상태, 평온함
- **감성 인식**: 편안함, 안정감, 집중된 이완 상태

#### 베타파 (Beta, 16-30Hz)
- **특징**: 의식적 활동, 복잡한 계산 처리, 집중력 발휘 시 활성화
- **생리적 의미**: 일반적인 각성 상태, 논리적 사고
- **감성 인식**: 활동적 감정, 스트레스, 긴장 상태

#### 감마파 (Gamma, 32-50Hz)
- **특징**: 정서적으로 초조, 추리와 같은 고도의 인지 정보처리 시 활성화
- **생리적 의미**: 높은 수준의 인지 활동, 의식적 통합
- **감성 인식**: 강한 감정적 경험, 고도의 집중
- **특성**: 편안하고 안정된 상태일수록 활성화
- **생리적 의미**: 시각적 정보가 차단되면 개안 대비 알파 활성화
- **감성 인식**: 정서적 안정, 집중 상태의 주요 지표

#### 베타파 (Beta, 16-30Hz)
- **특징**: 의식적 활동, 복잡한 계산 처리, 집중력 발휘 시 활성화
- **생리적 의미**: 활발한 사고 활동, 문제 해결
- **감성 인식**: 인지 부하, 스트레스, 불안 상태 반영

#### 감마파 (Gamma, 32-50Hz)
- **특징**: 정서적 초조, 추리와 같은 고도의 인지 정보 처리 시 활성화
- **생리적 의미**: 고차원적 인지 기능, 의식적 인식
- **감성 인식**: 고도의 집중, 각성 상태

## 3. ERP (Event Related Potential) 분석

### ERP 신호 처리 과정

#### 1. 자극 기반 데이터 추출
```python
def extract_erp_epochs(eeg_data, stimulus_times, fs, pre_stim=0.2, post_stim=1.0):
    """
    자극 시점을 기준으로 ERP 에포크 추출
    
    Parameters:
    - eeg_data: 연속적인 뇌파 데이터
    - stimulus_times: 자극 제시 시점 (초 단위)
    - fs: 샘플링 주파수
    - pre_stim: 자극 전 시간 (초)
    - post_stim: 자극 후 시간 (초)
    """
    pre_samples = int(pre_stim * fs)
    post_samples = int(post_stim * fs)
    epoch_length = pre_samples + post_samples
    
    epochs = []
    
    for stim_time in stimulus_times:
        stim_sample = int(stim_time * fs)
        start_sample = stim_sample - pre_samples
        end_sample = stim_sample + post_samples
        
        if start_sample >= 0 and end_sample < len(eeg_data):
            epoch = eeg_data[start_sample:end_sample]
            epochs.append(epoch)
    
    return np.array(epochs)
```

#### 2. 평균화를 통한 패턴 추출
```python
def calculate_erp(epochs):
    """
    에포크들의 평균화를 통해 ERP 패턴 계산
    """
    # 모든 에포크의 평균 계산
    erp_pattern = np.mean(epochs, axis=0)
    
    # 표준 오차 계산
    erp_std = np.std(epochs, axis=0) / np.sqrt(len(epochs))
    
    return erp_pattern, erp_std
```

#### 3. Peak와 Valley 검출
```python
def detect_erp_components(erp_pattern, fs, pre_stim=0.2):
    """
    ERP 패턴에서 주요 성분 검출
    """
    from scipy.signal import find_peaks
    
    # 시간 축 생성
    time_axis = np.linspace(-pre_stim, len(erp_pattern)/fs - pre_stim, len(erp_pattern))
    
    # Peak 검출 (양의 성분)
    peaks, peak_properties = find_peaks(erp_pattern, height=0, distance=int(0.05*fs))
    
    # Valley 검출 (음의 성분)
    valleys, valley_properties = find_peaks(-erp_pattern, height=0, distance=int(0.05*fs))
    
    components = {
        'peaks': {
            'indices': peaks,
            'times': time_axis[peaks],
            'amplitudes': erp_pattern[peaks]
        },
        'valleys': {
            'indices': valleys,
            'times': time_axis[valleys],
            'amplitudes': erp_pattern[valleys]
        }
    }
    
    return components

def calculate_component_features(components):
    """
    ERP 성분의 특징값 계산
    """
    features = {}
    
    # P300 성분 검출 (300ms 근처의 양의 peak)
    p300_candidates = [(i, t, a) for i, t, a in zip(
        components['peaks']['indices'],
        components['peaks']['times'],
        components['peaks']['amplitudes']
    ) if 0.25 <= t <= 0.4]  # 250-400ms 범위
    
    if p300_candidates:
        p300_idx, p300_time, p300_amp = max(p300_candidates, key=lambda x: x[2])
        features['P300_latency'] = p300_time * 1000  # ms 단위
        features['P300_amplitude'] = p300_amp
    
    # N200 성분 검출 (200ms 근처의 음의 peak)
    n200_candidates = [(i, t, a) for i, t, a in zip(
        components['valleys']['indices'],
        components['valleys']['times'],
        components['valleys']['amplitudes']
    ) if 0.15 <= t <= 0.25]  # 150-250ms 범위
    
    if n200_candidates:
        n200_idx, n200_time, n200_amp = min(n200_candidates, key=lambda x: x[2])
        features['N200_latency'] = n200_time * 1000  # ms 단위
        features['N200_amplitude'] = n200_amp
    
    return features
```

#### 4. 구간별 평균 진폭 계산
```python
def calculate_time_window_features(erp_pattern, fs, pre_stim=0.2, window_size=0.1):
    """
    100ms 단위 구간별 평균 진폭 값 계산
    """
    time_axis = np.linspace(-pre_stim, len(erp_pattern)/fs - pre_stim, len(erp_pattern))
    window_samples = int(window_size * fs)
    
    features = {}
    
    # 시간 구간별 평균 계산
    for start_time in np.arange(0, 1.0, window_size):  # 0-1초, 100ms 구간
        end_time = start_time + window_size
        
        # 해당 시간 구간의 샘플 인덱스
        start_idx = np.argmin(np.abs(time_axis - start_time))
        end_idx = np.argmin(np.abs(time_axis - end_time))
        
        # 구간 평균 진폭
        window_mean = np.mean(erp_pattern[start_idx:end_idx])
        features[f'mean_{int(start_time*1000)}_{int(end_time*1000)}ms'] = window_mean
    
    return features
```

### 사건 관련 전위 (Event Related Potential, ERP) 분석

#### ERP의 개념과 특징

**정의**: 특정 감각, 인지, 운동 사건과 시간적으로 연결된 뇌파의 변화
- **시간 고정**: 특정 자극이나 사건에 시간적으로 고정됨
- **평균화**: 여러 시행의 평균을 통해 노이즈 감소
- **성분 분석**: 특정 시간 구간의 양성 또는 음성 전위 성분 분석

#### 주요 ERP 성분

**P300 (P3) 성분**:
- **지연시간**: 자극 후 약 300ms
- **의미**: 주의 집중, 의사결정 과정
- **응용**: 거짓말 탐지기, 인지 부하 측정

```python
def analyze_p300(eeg_epochs, sampling_rate=1000):
    """
    P300 성분 분석
    """
    # 자극 후 200-400ms 구간에서 최대값 찾기
    start_idx = int(0.2 * sampling_rate)  # 200ms
    end_idx = int(0.4 * sampling_rate)    # 400ms
    
    p300_window = eeg_epochs[:, start_idx:end_idx]
    p300_amplitude = np.max(p300_window, axis=1)
    p300_latency = np.argmax(p300_window, axis=1) * (1000/sampling_rate) + 200
    
    return p300_amplitude, p300_latency
```

**N400 성분**:
- **지연시간**: 자극 후 약 400ms
- **의미**: 의미론적 처리, 언어 이해
- **응용**: 언어 인지 연구, 감정적 맥락 분석

**MMN (Mismatch Negativity)**:
- **지연시간**: 자극 후 100-250ms
- **의미**: 청각적 변화 탐지
- **응용**: 청각 인지 기능 평가

#### SSVEP (Steady-State Visual Evoked Potential) 분석

**정의**: 일정한 주파수로 깜빡이는 시각 자극에 대한 뇌의 반응

```python
def ssvep_analysis(eeg_signal, stimulus_freq, fs=1000):
    """
    SSVEP 신호 분석
    """
    # FFT를 통한 주파수 분석
    fft_values = np.fft.fft(eeg_signal)
    frequencies = np.fft.fftfreq(len(eeg_signal), 1/fs)
    
    # 자극 주파수에서의 파워 추출
    stimulus_idx = np.argmin(np.abs(frequencies - stimulus_freq))
    ssvep_power = np.abs(fft_values[stimulus_idx]) ** 2
    
    # 하모닉 성분도 고려
    harmonic_powers = []
    for harmonic in [2, 3, 4]:
        harmonic_freq = stimulus_freq * harmonic
        harmonic_idx = np.argmin(np.abs(frequencies - harmonic_freq))
        harmonic_power = np.abs(fft_values[harmonic_idx]) ** 2
        harmonic_powers.append(harmonic_power)
    
    return ssvep_power, harmonic_powers
```

**응용 분야**:
- Brain-Computer Interface (BCI)
- 주의 집중 측정
- 시각적 인지 부하 평가

## 4. SSVEP (Steady State Visually Evoked Potentials)

### SSVEP 원리 및 구현

#### 1. SSVEP 신호 특성
- **정의**: Hz 단위의 시각적 자극에 뇌파가 동기화되는 현상
- **예시**: 초당 13Hz로 깜빡이는 시각적 자극을 보면 뇌파가 13Hz에서 활성화
- **응용**: BCI 시스템의 입력 신호로 활용

#### 2. SSVEP 검출 구현
```python
def detect_ssvep(eeg_signal, fs, analysis_time=2.0):
    """
    SSVEP 신호에서 주파수 검출
    
    Parameters:
    - eeg_signal: 뇌파 신호
    - fs: 샘플링 주파수
    - analysis_time: 분석 시간 (초)
    """
    # 분석할 샘플 수
    n_samples = int(analysis_time * fs)
    
    if len(eeg_signal) < n_samples:
        raise ValueError("신호 길이가 분석 시간보다 짧습니다.")
    
    # 최근 분석 시간만큼의 신호 추출
    analysis_signal = eeg_signal[-n_samples:]
    
    # FFT를 통한 주파수 분석
    fft_values = np.fft.fft(analysis_signal)
    frequencies = np.fft.fftfreq(n_samples, 1/fs)
    
    # 양의 주파수만 사용
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    power_spectrum = np.abs(fft_values[positive_freq_idx]) ** 2
    
    # 최대 파워를 가진 주파수 검출
    max_power_idx = np.argmax(power_spectrum)
    detected_frequency = frequencies[max_power_idx]
    
    # SSVEP 주파수 계산 공식
    # Max Hz = 검출 인덱스 * (1/analysis_time)
    ssvep_frequency = max_power_idx * (1/analysis_time)
    
    return {
        'detected_frequency': detected_frequency,
        'ssvep_frequency': ssvep_frequency,
        'max_power': power_spectrum[max_power_idx],
        'confidence': power_spectrum[max_power_idx] / np.mean(power_spectrum)
    }

def create_ssvep_classifier(target_frequencies):
    """
    SSVEP 기반 BCI 분류기 생성
    
    Parameters:
    - target_frequencies: 목표 자극 주파수 리스트 (예: [8, 10, 12, 15])
    """
    class SSVEPClassifier:
        def __init__(self, frequencies):
            self.target_frequencies = frequencies
            
        def classify(self, eeg_signal, fs, threshold=2.0):
            """
            SSVEP 신호 분류
            
            Returns:
            - classified_frequency: 분류된 주파수 (또는 None)
            - confidence: 신뢰도
            """
            result = detect_ssvep(eeg_signal, fs)
            detected_freq = result['detected_frequency']
            confidence = result['confidence']
            
            # 임계값 확인
            if confidence < threshold:
                return None, confidence
            
            # 가장 가까운 목표 주파수 찾기
            distances = [abs(detected_freq - freq) for freq in self.target_frequencies]
            min_distance_idx = np.argmin(distances)
            
            # 허용 오차 내에 있는지 확인 (±0.5Hz)
            if distances[min_distance_idx] <= 0.5:
                return self.target_frequencies[min_distance_idx], confidence
            else:
                return None, confidence
    
    return SSVEPClassifier(target_frequencies)
```

## 5. 심전도 개요 및 측정 원리

### 심전도(ECG) 기본 개념

#### 정의 및 특성
- **심전도**: 심장 활동의 수축성 전기적 징후를 보여주는 신호
- **주요 지표**: 분당 Heart Rate (BPM, Beats Per Minute)
- **신호 특성**: 심장의 심방과 심실이 수축과 확장을 통해 발생되는 전기적 활동

#### 심박수 분류
- **빈맥 (Tachycardia)**: 심장 박동수 분당 100회 이상
- **정상 맥박**: 성인 기준 평균 60-80회/분
- **서맥 (Bradycardia)**: 심장 박동수 분당 60회 이하

### ECG 신호 구성 요소

#### 주요 파형 성분

**P wave**:
- **지속시간**: 약 80ms
- **원인**: 심실 수축에 의해 발생
- **특성**: 느린 수축, 심전도의 전위 세기 약 0.1-0.2mV
- **의미**: 심방 탈분극

**PQ segment**:
- **의미**: AV 노드가 흥분한 후 심실이 수축하기 전까지의 지연 시간
- **생리적 역할**: 심방에서 심실로의 전기 전도 지연

**QRS wave**:
- **원인**: 푸르키네 섬유 수축 후 발생
- **특성**: 1mV의 큰 진폭을 보임
- **의미**: 심실 탈분극, ECG에서 가장 뚜렷한 신호

**ST segment**:
- **원인**: 고원전압에 의해 발생
- **의미**: 심실 재분극의 초기 단계

**T wave**:
- **원인**: 심실의 탈분극 현상으로 나타남
- **특성**: 항상 보이지는 않으며, 0.1-0.3mV의 낮은 파형
- **의미**: 심실 재분극 완료

## 6. 심전도 신호 처리

### ECG 특징 추출 파이프라인

#### 1단계: 원신호 수집 및 필터링
```python
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks

def preprocess_ecg(ecg_signal, fs):
    """
    ECG 신호 전처리
    """
    # 밴드패스 필터 (0.5-40Hz)
    nyquist = fs / 2
    low_freq = 0.5 / nyquist
    high_freq = 40.0 / nyquist
    
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    filtered_ecg = signal.filtfilt(b, a, ecg_signal)
    
    return filtered_ecg
```

#### 2단계: QRS 성분 검출
```python
def detect_qrs_peaks(ecg_signal, fs):
    """
    QRS 복합체에서 R-peak 검출
    """
    # 미분 기반 QRS 검출
    diff_ecg = np.diff(ecg_signal)
    squared_ecg = diff_ecg ** 2
    
    # 이동 평균 필터
    window_size = int(0.08 * fs)  # 80ms window
    moving_avg = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
    
    # Peak 검출
    # 최소 거리: 0.6초 (분당 100회 제한)
    min_distance = int(0.6 * fs)
    
    # 적응적 임계값 설정
    threshold = 0.4 * np.max(moving_avg)
    
    peaks, properties = find_peaks(moving_avg, 
                                   height=threshold, 
                                   distance=min_distance)
    
    # 원본 신호에서 정확한 R-peak 위치 찾기
    r_peaks = []
    for peak in peaks:
        # Peak 주변에서 최대값 찾기
        search_window = int(0.05 * fs)  # ±50ms
        start_idx = max(0, peak - search_window)
        end_idx = min(len(ecg_signal), peak + search_window)
        
        local_max_idx = np.argmax(ecg_signal[start_idx:end_idx])
        r_peak = start_idx + local_max_idx
        r_peaks.append(r_peak)
    
    return np.array(r_peaks)
```

#### 3단계: 심박수 및 변이성 계산
```python
def calculate_hrv_features(r_peaks, fs):
    """
    Heart Rate Variability (HRV) 특징 계산
    """
    # R-R 간격 계산 (ms 단위)
    rr_intervals = np.diff(r_peaks) / fs * 1000
    
    # 기본 통계량
    features = {}
    
    # 평균 심박수 (BPM)
    mean_rr = np.mean(rr_intervals)
    features['heart_rate'] = 60000 / mean_rr  # BPM
    
    # SDNN: R-R 간격의 표준편차
    features['SDNN'] = np.std(rr_intervals)
    
    # RMSSD: 연속된 R-R 간격 차이의 제곱근 평균
    rr_diff = np.diff(rr_intervals)
    features['RMSSD'] = np.sqrt(np.mean(rr_diff ** 2))
    
    # pNN50: 연속된 R-R 간격 차이가 50ms를 초과하는 비율
    nn50_count = np.sum(np.abs(rr_diff) > 50)
    features['pNN50'] = (nn50_count / len(rr_diff)) * 100
    
    # 삼각지수: RR 간격 히스토그램의 삼각 보간
    hist, bin_edges = np.histogram(rr_intervals, bins=50)
    features['triangular_index'] = len(rr_intervals) / np.max(hist)
    
    return features, rr_intervals
```

### HRV (Heart Rate Variability) 주파수 분석

#### 1. 주파수 영역 변환
```python
def analyze_hrv_frequency(rr_intervals, method='welch'):
    """
    HRV 주파수 영역 분석
    """
    # R-R 간격을 균등 간격으로 보간
    time_rr = np.cumsum(rr_intervals) / 1000  # 초 단위
    fs_interp = 4  # 4Hz 보간
    
    # 보간 시간 축 생성
    time_interp = np.arange(0, time_rr[-1], 1/fs_interp)
    
    # 스플라인 보간
    from scipy import interpolate
    f_interp = interpolate.interp1d(time_rr, rr_intervals, kind='cubic')
    rr_interp = f_interp(time_interp)
    
    # 파워 스펙트럼 계산
    if method == 'welch':
        frequencies, psd = signal.welch(rr_interp, fs_interp, nperseg=256)
    elif method == 'fft':
        fft_vals = np.fft.fft(rr_interp)
        frequencies = np.fft.fftfreq(len(rr_interp), 1/fs_interp)
        psd = np.abs(fft_vals) ** 2
        
        # 양의 주파수만 사용
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        psd = psd[positive_freq_idx]
    
    return frequencies, psd

def extract_hrv_frequency_features(frequencies, psd):
    """
    HRV 주파수 대역별 파워 추출
    """
    # 주파수 대역 정의
    vlf_band = (0.0033, 0.04)  # Very Low Frequency
    lf_band = (0.04, 0.15)     # Low Frequency  
    hf_band = (0.15, 0.4)      # High Frequency
    
    # 각 대역의 파워 계산
    def get_band_power(freq_range):
        band_idx = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        if np.any(band_idx):
            return np.trapz(psd[band_idx], frequencies[band_idx])
        return 0
    
    vlf_power = get_band_power(vlf_band)
    lf_power = get_band_power(lf_band)
    hf_power = get_band_power(hf_band)
    
    total_power = vlf_power + lf_power + hf_power
    
    features = {
        'VLF': vlf_power,
        'LF': lf_power,
        'HF': hf_power,
        'Total_Power': total_power,
        'LF_HF_Ratio': lf_power / hf_power if hf_power > 0 else 0,
        'LF_norm': lf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0,
        'HF_norm': hf_power / (lf_power + hf_power) if (lf_power + hf_power) > 0 else 0
    }
    
    return features
```

### HRV 주파수 대역의 생리적 의미

#### VLF (Very Low Frequency, 0.0033-0.04 Hz)

**생리적 의미**:
- 교감신경계 반영
- 체온 조절, 혈압-체액전해질 조절계와 관련
- 장기간의 생리적 조절 기능 반영

**임상적 의미**:
- 수면 중 무호흡증(apnea), 호흡 정지(respiratory arrest) 시 활성화
- 저산소증(hypoxemia) 상태에서 해당 영역 파워 값 증가
- 피로(fatigue), 부정맥(arrhythmias) 환자에서 낮은 VLF power 관찰

#### LF (Low Frequency, 0.04-0.15 Hz)

**생리적 의미**:
- 교감신경과 부교감신경의 활성화를 모두 반영
- 부교감 신경계의 영향: 분당 호흡률 7회 이하 또는 깊은 호흡 시
- 혈압 조절 시스템과 관련 (baroreflex)

**특성**:
- 깊은 이완 상태, 수면 시 자연스럽게 발생
- 0.1 Hz 근처에서 특히 활성화
- 스트레스 상태에서 증가하는 경향

#### HF (High Frequency, 0.15-0.4 Hz)

**생리적 의미**:
- 부교감 신경계와 미주신경의 활성화 반영
- 호흡 주기(respiratory cycle)와 밀접한 관련
- RSA(respiratory sinus arrhythmia)와 연관되어 respiratory band로도 불림

**특성**:
- 호흡에 의한 심박수 변이 반영
- 이완 상태, 명상 시 증가
- 스트레스, 불안 상태에서 감소

## 7. 감성 인식을 위한 뇌-심장 신호 융합

### 멀티모달 특징 추출

```python
class BrainHeartEmotionRecognizer:
    def __init__(self, eeg_channels=8, ecg_channels=1):
        self.eeg_channels = eeg_channels
        self.ecg_channels = ecg_channels
        
    def extract_eeg_features(self, eeg_data, fs):
        """
        뇌파에서 감성 관련 특징 추출
        """
        features = {}
        
        for ch in range(self.eeg_channels):
            channel_data = eeg_data[ch]
            
            # 주파수 대역별 파워 추출
            frequencies, power_spectrum = frequency_analysis(channel_data, fs)
            band_powers = extract_frequency_bands(frequencies, power_spectrum)
            
            # 채널별 특징 저장
            for band, power in band_powers.items():
                features[f'eeg_ch{ch}_{band}'] = power
            
            # 비대칭성 분석 (좌우 뇌 불균형)
            if ch % 2 == 1 and ch > 0:  # 홀수 채널 (우측)
                left_ch = ch - 1
                right_ch = ch
                
                # 알파 비대칭성 (좌우 알파파 차이)
                left_alpha = features[f'eeg_ch{left_ch}_alpha']
                right_alpha = features[f'eeg_ch{right_ch}_alpha']
                features[f'alpha_asymmetry_ch{left_ch}_{right_ch}'] = (right_alpha - left_alpha) / (right_alpha + left_alpha)
        
        # 감정 상태 추정을 위한 조합 특징
        features['arousal_index'] = features.get('eeg_ch0_beta', 0) / features.get('eeg_ch0_alpha', 1)
        features['valence_index'] = features.get('alpha_asymmetry_ch0_1', 0)
        
        return features
    
    def extract_ecg_features(self, ecg_data, fs):
        """
        심전도에서 감성 관련 특징 추출
        """
        # ECG 전처리
        filtered_ecg = preprocess_ecg(ecg_data, fs)
        
        # R-peak 검출
        r_peaks = detect_qrs_peaks(filtered_ecg, fs)
        
        # HRV 특징 추출
        hrv_features, rr_intervals = calculate_hrv_features(r_peaks, fs)
        
        # 주파수 영역 HRV 분석
        frequencies, psd = analyze_hrv_frequency(rr_intervals)
        freq_features = extract_hrv_frequency_features(frequencies, psd)
        
        # 감성 관련 지표 계산
        emotion_features = {
            'stress_index': freq_features['LF_HF_Ratio'],  # 스트레스 지표
            'relaxation_index': freq_features['HF_norm'],   # 이완 지표
            'autonomic_balance': freq_features['LF_norm'] - freq_features['HF_norm']  # 자율신경계 균형
        }
        
        # 모든 특징 통합
        all_features = {**hrv_features, **freq_features, **emotion_features}
        
        return all_features
    
    def fuse_features(self, eeg_features, ecg_features):
        """
        뇌파와 심전도 특징 융합
        """
        # 정규화
        def normalize_features(features):
            normalized = {}
            for key, value in features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    normalized[key] = (value - np.mean(list(features.values()))) / (np.std(list(features.values())) + 1e-8)
                else:
                    normalized[key] = 0
            return normalized
        
        norm_eeg = normalize_features(eeg_features)
        norm_ecg = normalize_features(ecg_features)
        
        # 융합된 특징 벡터 생성
        fused_features = {**norm_eeg, **norm_ecg}
        
        # 상호작용 특징 (Cross-modal features)
        fused_features['brain_heart_coherence'] = norm_eeg.get('arousal_index', 0) * norm_ecg.get('stress_index', 0)
        fused_features['cognitive_load'] = (norm_eeg.get('eeg_ch0_beta', 0) + norm_ecg.get('LF_HF_Ratio', 0)) / 2
        
        return fused_features

### 실시간 감성 인식 시스템

class RealTimeEmotionRecognition:
    def __init__(self, window_size=5, overlap=0.5):
        """
        실시간 감성 인식 시스템
        
        Parameters:
        - window_size: 분석 윈도우 크기 (초)
        - overlap: 윈도우 겹침 비율 (0-1)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.brain_heart_recognizer = BrainHeartEmotionRecognizer()
        
        # 순환 버퍼 초기화
        self.eeg_buffer = None
        self.ecg_buffer = None
        self.buffer_size = None
        
        # 감정 분류 모델 (사전 훈련된 모델 가정)
        self.emotion_classifier = None
        
        # 감정 상태 히스토리
        self.emotion_history = []
        self.confidence_threshold = 0.7
        
    def initialize_buffers(self, fs, num_eeg_channels=8):
        """
        데이터 버퍼 초기화
        """
        self.fs = fs
        self.buffer_size = int(self.window_size * fs)
        self.step_size = int(self.buffer_size * (1 - self.overlap))
        
        self.eeg_buffer = np.zeros((num_eeg_channels, self.buffer_size))
        self.ecg_buffer = np.zeros(self.buffer_size)
        
        self.buffer_pointer = 0
        
    def update_buffers(self, new_eeg_data, new_ecg_data):
        """
        새로운 데이터로 버퍼 업데이트
        """
        batch_size = new_eeg_data.shape[1]
        
        if self.buffer_pointer + batch_size <= self.buffer_size:
            # 버퍼에 공간이 충분한 경우
            self.eeg_buffer[:, self.buffer_pointer:self.buffer_pointer + batch_size] = new_eeg_data
            self.ecg_buffer[self.buffer_pointer:self.buffer_pointer + batch_size] = new_ecg_data
            self.buffer_pointer += batch_size
        else:
            # 버퍼가 가득 찬 경우, 순환 버퍼 방식으로 업데이트
            remaining_space = self.buffer_size - self.buffer_pointer
            
            # 남은 공간 채우기
            if remaining_space > 0:
                self.eeg_buffer[:, self.buffer_pointer:] = new_eeg_data[:, :remaining_space]
                self.ecg_buffer[self.buffer_pointer:] = new_ecg_data[:remaining_space]
            
            # 버퍼 앞부분에 나머지 데이터 추가
            overflow = batch_size - remaining_space
            if overflow > 0:
                self.eeg_buffer[:, :overflow] = new_eeg_data[:, remaining_space:]
                self.ecg_buffer[:overflow] = new_ecg_data[remaining_space:]
            
            self.buffer_pointer = overflow if overflow > 0 else 0
    
    def is_ready_for_analysis(self):
        """
        분석 준비 상태 확인
        """
        return self.buffer_pointer >= self.step_size or np.any(self.eeg_buffer != 0)
    
    def classify_emotion(self, features):
        """
        특징 벡터를 기반으로 감정 분류
        """
        # 실제로는 사전 훈련된 머신러닝 모델 사용
        # 여기서는 간단한 규칙 기반 분류 예시
        
        arousal = features.get('arousal_index', 0)
        valence = features.get('valence_index', 0)
        stress = features.get('stress_index', 1)
        relaxation = features.get('relaxation_index', 0)
        
        # 2차원 감정 모델 (Russell's Circumplex Model)
        if arousal > 0 and valence > 0:
            emotion = 'happy'
            confidence = min(arousal + valence, 1.0)
        elif arousal > 0 and valence < 0:
            emotion = 'angry'
            confidence = min(arousal - valence, 1.0)
        elif arousal < 0 and valence > 0:
            emotion = 'calm'
            confidence = min(-arousal + valence, 1.0)
        elif arousal < 0 and valence < 0:
            emotion = 'sad'
            confidence = min(-arousal - valence, 1.0)
        else:
            emotion = 'neutral'
            confidence = 0.5
        
        # 스트레스 상태 추가 고려
        if stress > 2.0:
            emotion = 'stressed'
            confidence = min(stress / 3.0, 1.0)
        elif relaxation > 0.7:
            emotion = 'relaxed'
            confidence = relaxation
            
        return emotion, abs(confidence)
    
    def process_real_time(self, new_eeg_data, new_ecg_data):
        """
        실시간 데이터 처리 및 감정 인식
        """
        # 버퍼 업데이트
        self.update_buffers(new_eeg_data, new_ecg_data)
        
        # 분석 준비 확인
        if not self.is_ready_for_analysis():
            return None
        
        try:
            # 특징 추출
            eeg_features = self.brain_heart_recognizer.extract_eeg_features(self.eeg_buffer, self.fs)
            ecg_features = self.brain_heart_recognizer.extract_ecg_features(self.ecg_buffer, self.fs)
            
            # 특징 융합
            fused_features = self.brain_heart_recognizer.fuse_features(eeg_features, ecg_features)
            
            # 감정 분류
            emotion, confidence = self.classify_emotion(fused_features)
            
            # 신뢰도 기반 필터링
            if confidence >= self.confidence_threshold:
                result = {
                    'emotion': emotion,
                    'confidence': confidence,
                    'timestamp': np.datetime64('now'),
                    'features': fused_features
                }
                
                # 감정 히스토리 업데이트
                self.emotion_history.append(result)
                if len(self.emotion_history) > 10:  # 최근 10개 결과만 유지
                    self.emotion_history.pop(0)
                
                return result
            
        except Exception as e:
            print(f"실시간 처리 중 오류 발생: {e}")
            return None
        
        return None
    
    def get_stable_emotion(self, window=5):
        """
        최근 window개 결과의 안정적인 감정 상태 반환
        """
        if len(self.emotion_history) < window:
            return None
        
        recent_emotions = [result['emotion'] for result in self.emotion_history[-window:]]
        recent_confidences = [result['confidence'] for result in self.emotion_history[-window:]]
        
        # 가장 빈번한 감정 찾기
        from collections import Counter
        emotion_counts = Counter(recent_emotions)
        most_common_emotion = emotion_counts.most_common(1)[0][0]
        
        # 해당 감정의 평균 신뢰도
        emotion_confidences = [conf for emotion, conf in zip(recent_emotions, recent_confidences) 
                             if emotion == most_common_emotion]
        avg_confidence = np.mean(emotion_confidences)
        
        return {
            'stable_emotion': most_common_emotion,
            'stability': len(emotion_confidences) / window,
            'confidence': avg_confidence
        }

### 실사용 예시

def emotion_recognition_demo():
    """
    실시간 감성 인식 시스템 데모
    """
    # 시스템 초기화
    recognizer = RealTimeEmotionRecognition(window_size=5, overlap=0.5)
    recognizer.initialize_buffers(fs=1000, num_eeg_channels=8)
    
    # 시뮬레이션 데이터로 테스트
    duration = 60  # 60초 시뮬레이션
    batch_duration = 0.1  # 100ms 배치
    
    print("실시간 감성 인식 시스템 시작...")
    
    for t in np.arange(0, duration, batch_duration):
        # 시뮬레이션 데이터 생성 (실제로는 하드웨어에서 받음)
        batch_samples = int(batch_duration * 1000)
        
        # 가상의 뇌파 데이터 (8채널)
        sim_eeg = np.random.randn(8, batch_samples) * 0.1
        
        # 감정 시뮬레이션: 시간에 따라 다른 감정 패턴
        if t < 20:  # 처음 20초: 평온한 상태
            sim_eeg[0] += np.sin(2 * np.pi * 10 * np.linspace(t, t + batch_duration, batch_samples))  # 알파파
        elif t < 40:  # 20-40초: 스트레스 상태
            sim_eeg[0] += np.sin(2 * np.pi * 20 * np.linspace(t, t + batch_duration, batch_samples))  # 베타파
        else:  # 40-60초: 이완 상태
            sim_eeg[0] += np.sin(2 * np.pi * 8 * np.linspace(t, t + batch_duration, batch_samples))   # 느린 알파파
        
        # 가상의 심전도 데이터
        heart_rate = 70 if t < 20 else (90 if t < 40 else 60)  # 상황별 심박수
        sim_ecg = np.sin(2 * np.pi * (heart_rate/60) * np.linspace(t, t + batch_duration, batch_samples))
        sim_ecg += np.random.randn(batch_samples) * 0.05  # 노이즈 추가
        
        # 실시간 처리
        result = recognizer.process_real_time(sim_eeg, sim_ecg)
        
        if result:
            print(f"Time: {t:.1f}s - Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
            
            # 안정적인 감정 상태 확인
            stable = recognizer.get_stable_emotion()
            if stable and stable['stability'] > 0.6:
                print(f"  → Stable emotion: {stable['stable_emotion']} (stability: {stable['stability']:.2f})")

# 데모 실행
if __name__ == "__main__":
    emotion_recognition_demo()

## 8. 응용 분야 및 활용 사례

### 8.1 헬스케어 및 의료 분야

#### 정신건강 모니터링
- **우울증 및 불안장애 조기 발견**: EEG 알파파 비대칭성과 HRV 분석을 통한 정신건강 상태 평가
- **스트레스 관리**: 실시간 스트레스 수준 모니터링 및 개입 시점 알림
- **수면 질 평가**: 뇌파와 심박변이성을 통한 수면 단계 분석 및 수면 질 개선

#### 인지 기능 평가
- **치매 조기 진단**: ERP 성분 분석을 통한 인지 기능 저하 탐지
- **주의력 결핍 장애(ADHD) 진단**: 뇌파 주파수 분석을 통한 주의집중력 평가
- **뇌졸중 환자 재활**: 뇌파 기반 신경가소성 모니터링

```python
def healthcare_monitoring_system():
    """
    헬스케어용 감정 모니터링 시스템
    """
    class HealthcareEmotionMonitor:
        def __init__(self):
            self.baseline_established = False
            self.personal_baseline = {}
            self.alert_thresholds = {
                'stress_level': 0.8,
                'depression_risk': 0.7,
                'anxiety_level': 0.75
            }
            
        def establish_baseline(self, eeg_data, ecg_data, fs, duration_days=7):
            """
            개인별 기준선 설정 (7일간의 데이터 수집)
            """
            # 일주일간의 다양한 상황에서 데이터 수집
            # 여기서는 시뮬레이션
            
            recognizer = BrainHeartEmotionRecognizer()
            daily_features = []
            
            for day in range(duration_days):
                # 하루 데이터 분석
                eeg_features = recognizer.extract_eeg_features(eeg_data, fs)
                ecg_features = recognizer.extract_ecg_features(ecg_data, fs)
                
                daily_features.append({**eeg_features, **ecg_features})
            
            # 기준선 계산 (평균 및 표준편차)
            feature_names = daily_features[0].keys()
            for feature in feature_names:
                values = [day_data[feature] for day_data in daily_features]
                self.personal_baseline[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range': (np.min(values), np.max(values))
                }
            
            self.baseline_established = True
            print("개인별 기준선 설정 완료")
            
        def assess_mental_health_risk(self, current_features):
            """
            현재 상태의 정신건강 위험도 평가
            """
            if not self.baseline_established:
                return {"error": "기준선이 설정되지 않았습니다"}
            
            risk_scores = {}
            
            # 우울증 위험도 (알파파 비대칭성 기반)
            alpha_asym = current_features.get('alpha_asymmetry_ch0_1', 0)
            baseline_asym = self.personal_baseline.get('alpha_asymmetry_ch0_1', {})
            
            if baseline_asym:
                asym_z_score = (alpha_asym - baseline_asym['mean']) / (baseline_asym['std'] + 1e-8)
                risk_scores['depression_risk'] = max(0, min(1, (asym_z_score + 2) / 4))  # -2~2를 0~1로 정규화
            
            # 불안 위험도 (베타파 활성도 기반)
            beta_power = current_features.get('eeg_ch0_beta', 0)
            baseline_beta = self.personal_baseline.get('eeg_ch0_beta', {})
            
            if baseline_beta:
                beta_z_score = (beta_power - baseline_beta['mean']) / (baseline_beta['std'] + 1e-8)
                risk_scores['anxiety_level'] = max(0, min(1, beta_z_score / 2 + 0.5))
            
            # 스트레스 수준 (HRV LF/HF 비율 기반)
            stress_index = current_features.get('stress_index', 1)
            baseline_stress = self.personal_baseline.get('stress_index', {})
            
            if baseline_stress:
                stress_z_score = (stress_index - baseline_stress['mean']) / (baseline_stress['std'] + 1e-8)
                risk_scores['stress_level'] = max(0, min(1, stress_z_score / 2 + 0.5))
            
            return risk_scores
        
        def generate_health_alerts(self, risk_scores):
            """
            건강 상태 기반 알림 생성
            """
            alerts = []
            
            for risk_type, score in risk_scores.items():
                threshold = self.alert_thresholds.get(risk_type, 0.7)
                
                if score > threshold:
                    severity = "높음" if score > 0.9 else "중간"
                    
                    if risk_type == 'depression_risk':
                        alerts.append({
                            'type': '우울 위험도',
                            'severity': severity,
                            'score': score,
                            'recommendation': '전문의 상담을 권장합니다. 규칙적인 운동과 충분한 수면을 취하세요.'
                        })
                    elif risk_type == 'anxiety_level':
                        alerts.append({
                            'type': '불안 수준',
                            'severity': severity,
                            'score': score,
                            'recommendation': '심호흡이나 명상을 통해 마음을 진정시키세요. 필요시 전문가와 상담하세요.'
                        })
                    elif risk_type == 'stress_level':
                        alerts.append({
                            'type': '스트레스 수준',
                            'severity': severity,
                            'score': score,
                            'recommendation': '휴식을 취하고 스트레스 요인을 파악해보세요. 이완 기법을 시도해보세요.'
                        })
            
            return alerts
    
    return HealthcareEmotionMonitor()
```

### 8.2 교육 및 학습 분야

#### 학습 상태 모니터링
- **집중도 측정**: 실시간 뇌파 분석을 통한 학습 집중도 평가
- **인지 부하 측정**: 학습 내용의 난이도와 개인의 인지 능력 매칭
- **학습 효율 최적화**: 개인별 최적 학습 시간대 및 방법 제안

#### 적응형 학습 시스템
```python
def adaptive_learning_system():
    """
    뇌파 기반 적응형 학습 시스템
    """
    class CognitiveLoadMonitor:
        def __init__(self):
            self.optimal_load_range = (0.4, 0.7)  # 최적 인지 부하 범위
            
        def assess_cognitive_load(self, eeg_features):
            """
            인지 부하 수준 평가
            """
            # 베타파/알파파 비율로 인지 부하 측정
            beta_power = eeg_features.get('eeg_ch0_beta', 0)
            alpha_power = eeg_features.get('eeg_ch0_alpha', 1)
            
            cognitive_load = beta_power / (alpha_power + 1e-8)
            
            # 0-1 범위로 정규화
            normalized_load = min(1.0, max(0.0, cognitive_load / 3.0))
            
            return normalized_load
        
        def recommend_difficulty_adjustment(self, current_load, target_load=0.55):
            """
            난이도 조정 권장사항
            """
            if current_load < self.optimal_load_range[0]:
                return {
                    'adjustment': 'increase_difficulty',
                    'message': '난이도를 높여 도전적인 내용을 제공하세요',
                    'factor': 1.2
                }
            elif current_load > self.optimal_load_range[1]:
                return {
                    'adjustment': 'decrease_difficulty',
                    'message': '난이도를 낮추고 복습 시간을 늘리세요',
                    'factor': 0.8
                }
            else:
                return {
                    'adjustment': 'maintain',
                    'message': '현재 난이도가 적절합니다',
                    'factor': 1.0
                }
    
    return CognitiveLoadMonitor()
```

### 8.3 게임 및 엔터테인먼트

#### 감정 반응형 게임
- **몰입도 측정**: 플레이어의 실시간 몰입 상태 분석
- **난이도 자동 조절**: 스트레스 수준에 따른 게임 난이도 실시간 조정
- **감정 기반 스토리텔링**: 플레이어의 감정 상태에 맞는 게임 스토리 전개

#### BCI 게임 인터페이스
```python
def bci_game_interface():
    """
    뇌파 기반 게임 인터페이스
    """
    class BCIGameController:
        def __init__(self):
            self.ssvep_frequencies = [8, 10, 12, 15]  # 4개 방향
            self.ssvep_classifier = create_ssvep_classifier(self.ssvep_frequencies)
            
        def detect_user_intention(self, eeg_data, fs):
            """
            사용자 의도 감지 (SSVEP 기반)
            """
            detected_freq, confidence = self.ssvep_classifier.classify(eeg_data, fs)
            
            if detected_freq:
                direction_map = {
                    8: 'up',
                    10: 'down', 
                    12: 'left',
                    15: 'right'
                }
                
                return {
                    'command': direction_map.get(detected_freq, 'none'),
                    'confidence': confidence
                }
            
            return {'command': 'none', 'confidence': 0}
        
        def assess_player_state(self, eeg_features, ecg_features):
            """
            플레이어 상태 평가
            """
            # 흥미도 (알파파 기반)
            alpha_power = eeg_features.get('eeg_ch0_alpha', 0)
            engagement = 1 / (1 + alpha_power)  # 알파파가 낮을수록 높은 집중
            
            # 스트레스 (심박변이성 기반)
            stress_level = ecg_features.get('stress_index', 1)
            
            # 몰입도 (감마파 기반)
            gamma_power = eeg_features.get('eeg_ch0_gamma', 0)
            flow_state = min(1.0, gamma_power / 0.5)
            
            return {
                'engagement': engagement,
                'stress': stress_level,
                'flow': flow_state,
                'recommendation': self._get_game_recommendation(engagement, stress_level, flow_state)
            }
        
        def _get_game_recommendation(self, engagement, stress, flow):
            """
            게임 조정 권장사항
            """
            if stress > 0.8:
                return "스트레스가 높습니다. 게임 속도를 늦추거나 휴식을 권장합니다."
            elif engagement < 0.3:
                return "흥미도가 낮습니다. 더 도전적인 요소를 추가하세요."
            elif flow > 0.7:
                return "몰입 상태입니다. 현재 설정을 유지하세요."
            else:
                return "균형잡힌 상태입니다."
    
    return BCIGameController()
```

### 8.4 인간-컴퓨터 상호작용 (HCI)

#### 감정 인식 기반 UI/UX
- **적응형 인터페이스**: 사용자 감정에 따른 UI 색상, 레이아웃 자동 조정
- **스마트 알림 시스템**: 스트레스 수준을 고려한 알림 타이밍 최적화
- **개인화된 사용자 경험**: 감정 패턴 학습을 통한 맞춤형 서비스 제공

#### 웰빙 및 정신건강 관리
- **명상 가이드 앱**: 실시간 뇌파 분석을 통한 명상 상태 피드백
- **스트레스 관리 시스템**: 직장에서의 실시간 스트레스 모니터링
- **수면 개선 앱**: 심박변이성 분석을 통한 수면 질 개선 가이드

## 9. 최신 연구 동향 및 기술 발전

### 9.1 딥러닝 기반 감성 인식

#### Transformer 기반 뇌파 분석
```python
def transformer_eeg_model():
    """
    Transformer 기반 뇌파 감성 인식 모델 (의사코드)
    """
    class EEGTransformer:
        def __init__(self, n_channels=8, seq_length=1000, d_model=512):
            self.n_channels = n_channels
            self.seq_length = seq_length
            self.d_model = d_model
            
        def positional_encoding(self, sequence):
            """
            시간적 위치 인코딩
            """
            # 뇌파 신호의 시간적 특성을 반영한 위치 인코딩
            pass
            
        def multi_head_attention(self, query, key, value):
            """
            채널 간 상호작용 모델링
            """
            # 뇌파 채널 간의 상호작용 학습
            pass
            
        def feed_forward(self, x):
            """
            비선형 변환
            """
            pass
        
        def predict_emotion(self, eeg_sequence):
            """
            감정 예측
            """
            # 전처리된 뇌파 시퀀스를 입력으로 받아 감정 분류
            pass
    
    return EEGTransformer()
```

### 9.2 연합학습 및 개인화

#### 개인화된 감성 인식 모델
- **전이 학습**: 일반 모델에서 개인별 특성으로 fine-tuning
- **연합학습**: 개인정보 보호하면서 모델 성능 향상
- **메타 러닝**: 새로운 사용자에 대한 빠른 적응

### 9.3 실시간 처리 최적화

#### 엣지 컴퓨팅 활용
```python
def edge_optimized_emotion_recognition():
    """
    엣지 디바이스 최적화된 감성 인식 시스템
    """
    class EdgeEmotionRecognizer:
        def __init__(self):
            self.lightweight_model = True
            self.compression_ratio = 0.1
            
        def quantize_model(self, model):
            """
            모델 양자화를 통한 크기 압축
            """
            # 32bit → 8bit 양자화
            pass
        
        def knowledge_distillation(self, teacher_model, student_model):
            """
            지식 증류를 통한 경량화
            """
            # 큰 모델의 지식을 작은 모델로 전달
            pass
        
        def real_time_inference(self, input_data):
            """
            실시간 추론 최적화
            """
            # CPU/GPU 하이브리드 처리
            # 캐싱 및 배치 처리 최적화
            pass
    
    return EdgeEmotionRecognizer()
```

## 10. 윤리적 고려사항 및 개인정보 보호

### 10.1 생체정보 보호

#### 데이터 암호화 및 익명화
```python
def privacy_preserving_system():
    """
    개인정보 보호 감성 인식 시스템
    """
    import hashlib
    from cryptography.fernet import Fernet
    
    class PrivacyPreservingEEG:
        def __init__(self):
            self.encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self.encryption_key)
            
        def anonymize_user_id(self, user_id):
            """
            사용자 ID 익명화
            """
            return hashlib.sha256(user_id.encode()).hexdigest()[:16]
        
        def encrypt_biometric_data(self, data):
            """
            생체 데이터 암호화
            """
            serialized_data = str(data).encode()
            encrypted_data = self.cipher.encrypt(serialized_data)
            return encrypted_data
        
        def differential_privacy_noise(self, data, epsilon=0.1):
            """
            차등 프라이버시 노이즈 추가
            """
            noise_scale = 1.0 / epsilon
            noise = np.random.laplace(0, noise_scale, data.shape)
            return data + noise
        
        def federated_feature_extraction(self, local_data):
            """
            연합학습 기반 특징 추출
            """
            # 로컬에서만 원본 데이터 처리
            # 집계 가능한 특징만 전송
            pass
    
    return PrivacyPreservingEEG()
```

### 10.2 사용자 동의 및 투명성

#### 설명 가능한 AI
- **특징 중요도 시각화**: 어떤 뇌파/심전도 특징이 감정 분류에 영향을 미쳤는지 설명
- **불확실성 정량화**: 모델 예측의 신뢰도 및 불확실성 제공
- **사용자 제어권**: 데이터 수집 및 처리에 대한 사용자 제어 옵션

## 11. 요약 및 정리

### 11.1 핵심 내용 요약

#### 뇌파(EEG) 기반 감성 인식
1. **신호 특성**: 다양한 주파수 대역(델타, 세타, 알파, 베타, 감마)별 감정 상태 반영
2. **공간적 정보**: 뇌 영역별(전두엽, 측두엽, 두정엽, 후두엽) 기능과 감정 처리
3. **시간적 특성**: ERP 성분을 통한 인지 과정 분석
4. **SSVEP 응용**: 뇌-컴퓨터 인터페이스 구현

#### 심전도(ECG) 기반 감성 인식
1. **HRV 분석**: 심박변이성을 통한 자율신경계 활동 평가
2. **주파수 영역**: VLF, LF, HF 대역별 생리적 의미
3. **스트레스 지표**: LF/HF 비율을 통한 스트레스 수준 측정
4. **개인화**: 개인별 기준선 설정을 통한 정확도 향상

#### 멀티모달 융합
1. **특징 융합**: 뇌파와 심전도 특징의 효과적 결합
2. **실시간 처리**: 순환 버퍼와 윈도우 기반 분석
3. **안정성**: 시간적 일관성을 고려한 감정 상태 판정
4. **개인화**: 사용자별 적응형 모델 구축

### 11.2 기술적 성과

#### 신호처리 기법
- **전처리**: 아티팩트 제거, 필터링, 정규화
- **특징 추출**: 시간/주파수 영역 특징, 통계적 특징
- **분류**: 기계학습/딥러닝 기반 감정 분류
- **후처리**: 시간적 평활화, 신뢰도 기반 필터링

#### 시스템 구현
- **실시간성**: 밀리초 단위 반응성
- **정확도**: 개인화를 통한 높은 분류 성능
- **안정성**: 노이즈 환경에서의 견고성
- **확장성**: 다양한 응용 분야 적용 가능

### 11.3 향후 발전 방향

#### 기술적 개선
1. **딥러닝 모델**: Transformer, CNN-LSTM 하이브리드 모델 개발
2. **엣지 컴퓨팅**: 실시간 처리를 위한 경량화 모델
3. **멀티모달 융합**: 더 많은 생체신호(EMG, GSR, 체온 등) 통합
4. **개인화**: 메타러닝 기반 빠른 개인 적응

#### 응용 분야 확장
1. **헬스케어**: 정밀의료, 원격진료, 예방의학
2. **교육**: 적응형 학습, 인지부하 최적화
3. **엔터테인먼트**: 몰입형 게임, VR/AR 경험
4. **산업**: 작업자 안전, 피로 관리, 생산성 향상

#### 사회적 영향
1. **정신건강**: 조기 진단 및 예방, 개인 맞춤 치료
2. **삶의 질**: 스트레스 관리, 웰빙 향상
3. **인간-기계 상호작용**: 더 자연스럽고 직관적인 인터페이스
4. **개인정보 보호**: 윤리적 AI 개발과 프라이버시 보장

뇌와 심장 기반 감성 인식 시스템은 인간의 감정을 객관적으로 측정하고 이해할 수 있는 혁신적인 기술입니다. 이 기술은 헬스케어부터 엔터테인먼트까지 다양한 분야에서 인간의 삶의 질을 향상시킬 수 있는 무한한 가능성을 가지고 있으며, 지속적인 연구와 개발을 통해 더욱 정확하고 실용적인 시스템으로 발전할 것입니다.
