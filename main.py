import streamlit as st
import json



def get_VECG(a):
    return ['dd', 'r']


def main(config : dict):
    st.title("Получение ВЭКГ")

    file_path = st.sidebar.file_uploader("Выберите файл .edf", type="edf")
    if file_path:
        st.write(f"Выбран файл: {file_path.name}")

    n_term_start = st.sidebar.number_input("Номер периода ЭКГ", value=config['n_term_start'], min_value=config['n_term_begin'], max_value=config['n_term_end'])
    button_pressed = st.sidebar.button(":red[Запуск]", key="launch_button",
                                       help="Нажмите, чтобы начать обработку данных", use_container_width=True)
    
    
    st.sidebar.markdown('---') 
    st.sidebar.markdown('### Выбор режимов:')
    plot_projections = st.sidebar.checkbox("Построение проекций ВЭКГ", value=config['plot_projections'])
    plot_3D = st.sidebar.checkbox("Построение 3D ВЭКГ", value=config['plot_3D'])
    show_ECG = st.sidebar.checkbox("Отображение ЭКГ сигналов", value=config['show_ECG'])
    predict_res = st.sidebar.checkbox("Результат СППР (болен/здоров)", value=config['predict_res'])
    count_qrst_angle = st.sidebar.checkbox("Расчет угла QRST", value=config['count_qrst_angle'])
    QRS_loop_area = st.sidebar.checkbox("Расчет площади QRS петли", value=config['QRS_loop_area'])
    T_loop_area = st.sidebar.checkbox("Расчет площади ST-T петли", value=config['T_loop_area'])

    st.sidebar.markdown('---') 
    st.sidebar.markdown('### Настройки:')
    mean_filter = st.sidebar.checkbox("Сглаживание петель", value=config['mean_filter'])
    filt = st.sidebar.checkbox("ФВЧ фильтрация ЭКГ сигналов", value=config['filt'])
    if filt:
        f_sreza = st.sidebar.number_input("Частота среза ФВЧ фильтра (в Гц)", value=config['f_sreza'])
    f_sampling = st.sidebar.number_input("Частота дискретизации (в Гц)", value=config['f_sampling'])
    logs = st.sidebar.checkbox("Показ логов обработки", value=config['logs'])  # Показать только при dev_mode

    st.sidebar.markdown('---') 
    
    if button_pressed:
        if file_path is not None:
            input_data = {
                "data_edf": file_path,
                "n_term_start": n_term_start,
                "filt": filt,
                "f_sreza": f_sreza if filt else None,
                "f_sampling": f_sampling,
                "plot_projections": plot_projections,
                "plot_3D": plot_3D,
                "show_ECG": show_ECG,
                "predict": predict_res,
                "count_qrst_angle": count_qrst_angle,
                "QRS_loop_area": QRS_loop_area,
                "T_loop_area": T_loop_area,
                "mean_filter": mean_filter,
                "logs": logs
            }

            # Получить ВЭКГ
            res = get_VECG(input_data)

            # Вывести результаты
            if isinstance(res, str):
                st.error(res)
            else:
                for result in res:
                    st.write(result)
        else:
            st.warning("Пожалуйста, загрузите файл .edf для обработки.")


if __name__ == "__main__":
    # Загрузка конфигурации программы:
    with open('configs/config.json', 'r') as json_file:
        config = json.load(json_file)
    main(config)

