import pandas as pd
import numpy as np
import altair as alt
import re
import warnings
import random
import streamlit as st

warnings.filterwarnings('ignore')
alt.data_transformers.enable('default', max_rows=None)

# data preprocessing

natural_amenity = pd.read_excel('natureamenity.xlsx')
unemployment_rate = pd.read_excel('UnemploymentRate.xlsx')
public_school = pd.read_csv('Public_Schools.csv')
mortality = pd.read_csv('mort.csv')
poverty_hhincome = pd.read_excel('poverty_household.xlsx')


def add_wrapping(line): # line is a string
    half = len(line.split()) // 2
    return ' '.join(line.split()[:half]) + '+' + ' '.join(line.split()[half:])


public_school = public_school[(public_school['ENROLLMENT'] > 0) & (public_school['FT_TEACHER'] > 0)]
public_school['ENROLLMENT_PER_TEACHER'] = public_school['ENROLLMENT'] / public_school['FT_TEACHER']
public_school['ENROLLMENT_RATE'] = public_school['ENROLLMENT'] / public_school['POPULATION'] * 100


school_type = ['ELEMENTARY', 'HIGH', 'MIDDLE']
public_school = public_school[public_school['LEVEL_'].isin(school_type)]
public_school_mn = public_school[public_school['STATE'] == 'MN']


unemployment_rate['Time'] = pd.to_datetime(unemployment_rate['Month'].apply(lambda x: str(x).replace('.', '')), 
                                           format='%Y%m', errors='coerce').dropna()
unemployment_rate_mn = pd.melt(unemployment_rate.drop(columns=['Month']), id_vars=['Time'], var_name='Type', value_name='UR')


mortality_mn = mortality[mortality['Location'].str.contains('Red Lake')]
mortality_years = ['Location', 'Category', 'Mortality Rate, 1980*', 'Mortality Rate, 1985*', 'Mortality Rate, 1990*', 
                   'Mortality Rate, 1995*', 'Mortality Rate, 2000*', 'Mortality Rate, 2005*', 'Mortality Rate, 2010*', 
                   'Mortality Rate, 2014*']
mortality_mn = mortality_mn[mortality_years]
mortality_mn = pd.melt(mortality_mn, id_vars=['Location', 'Category'], var_name='Year', value_name='MRP100k')
mortality_mn['MRP100k_log'] = mortality_mn['MRP100k'].apply(np.log)
mortality_mn['Year'] = mortality_mn['Year'].apply(lambda x: re.search('\d+', x).group(0))
mortality_mn_top5 = mortality_mn.sort_values(['Category', 'MRP100k'], ascending=True).groupby(['Year']).head(5)
mortality_mn_top5_list = list(mortality_mn_top5['Category'])
mortality_mn_top5['Category'] = mortality_mn_top5['Category'].apply(add_wrapping)
mortality_top5 = mortality[mortality['Category'].isin(mortality_mn_top5_list)]
mortality_top5_plot = mortality_top5.groupby(['Category']).mean().reset_index()
mortality_mn_plot = mortality[(mortality['Location'].str.contains('Red Lake')) & 
                              (mortality['Category'].isin(mortality_mn_top5_list))].groupby(['Category']).mean().reset_index()
mortality_top5_plot['Category'] = mortality_top5_plot['Category'].apply(add_wrapping)
mortality_mn_plot['Category'] = mortality_mn_plot['Category'].apply(add_wrapping)
year_list = ['1980', '1985', '1990', '1995', '2000', '2005', '2010', '2014']

# plotting

# line chart about unemployment rate
def plot_v1():
    s1_1 = alt.selection_interval(bind='scales', encodings=['x', 'y'])
    s1_2 = alt.selection_single(on='mouseover', empty='none')
    s1_3 = alt.selection_single(fields=['Time'], on='mouseover', empty='none')

    rule_1 = alt.Chart(unemployment_rate_mn, width=600, height=300).mark_rule(size=4, color='lightgray', opacity=0, strokeDash=[5, 5]).encode(
        x=alt.X('Time:T')
    ).add_selection(s1_2).encode(
        opacity=alt.condition(s1_2, alt.OpacityValue(1), alt.OpacityValue(0.00000001))
    )
    dots_1 = alt.Chart(unemployment_rate_mn).mark_circle(color='black', size=70, opacity=0).encode(
        x=alt.X('Time:T'),
        y=alt.Y('UR:Q'),
        tooltip=['Time', alt.Tooltip('Type', title='Data Source'), alt.Tooltip('UR', title='Unemployment Rate (%)')]
    ).add_selection(s1_3).encode(
        opacity=alt.condition(s1_3, alt.OpacityValue(1), alt.OpacityValue(0))
    )
    line_1 = alt.Chart(unemployment_rate_mn, title='Unemployment Rate 2011-2015', width=600, height=300).mark_line().encode(
        x=alt.X('Time:T', axis=alt.Axis(grid=False), title=None),
        y=alt.Y('UR:Q', title='Unemployment Rate (%)', axis=alt.Axis(grid=False), scale=alt.Scale(domain=[3,10])),
        color=alt.Color('Type:N', title='Data Source'),
        strokeDash=alt.condition(alt.datum.Type == 'National', alt.value([5, 5]), alt.value([0]))
    ).add_selection(s1_1).encode(
        color=alt.Color('Type:N', title='Data Source')
    )

    comp_1 = line_1 + rule_1 + dots_1

    return comp_1


# double scatter plot for poverty rate and household income
def plot_v2(state='MN'):
    s2_1 = alt.selection_single(on='click', empty='none')
    s2_2 = alt.selection_interval(bind='scales', encodings=['x'])
    s2_3 = alt.selection_interval(bind='scales', encodings=['y'])

    circle = alt.Chart(poverty_hhincome, width=400, height=250).transform_filter(
        (alt.datum['Postal Code'] != 'US') & (alt.datum['County FIPS Code'] == 0)
    ).mark_circle(size=20).encode(
        x=alt.X('Median Household Income:Q', axis=alt.Axis(grid=False), title='Median Household Income'),
        y=alt.Y('Poverty Percent, All Ages:Q', scale=alt.Scale(domain=[4,25]), 
                title='Poverty Percent (%)', axis=alt.Axis(grid=False)),
        size=alt.Size('Poverty Estimate, All Ages:Q', title='Poverty Population'),
        color=alt.condition(alt.datum.Name == 'Minnesota', alt.value('blue'), alt.value('brown')),
        tooltip=['Name', 'Median Household Income', alt.Tooltip('Poverty Percent, All Ages',  title='Poverty Percent, All Ages (%)'), alt.Tooltip('Poverty Estimate, All Ages', title='Poverty Population')]
    ).properties(
        title='Poverty Rate, Average House Income for each State'
    ).add_selection(s2_1, s2_2)

    xline = alt.Chart(poverty_hhincome).transform_filter(
        (alt.datum['Postal Code'] != 'US') & (alt.datum['County FIPS Code'] == 0)
    ).mark_rule(color='grey', opacity=0.5).encode(
        x='mean(Median Household Income)',
        strokeDash = alt.value([5,5])
    )
    xtext = xline.mark_text(align='left', y=20, dx=10, opacity=0.5).encode(
        text=alt.value('National Average')
    )

    xline_s = alt.Chart(poverty_hhincome).transform_filter(
        (alt.datum['Postal Code'] != 'US') & (alt.datum['County FIPS Code'] == 0)
    ).mark_rule(color='black', size=2, opacity=0, strokeDash=[5, 5]).encode(
        x='Median Household Income',
        opacity=alt.condition(s2_1, alt.OpacityValue(1), alt.OpacityValue(0))
    )

    yline = alt.Chart(poverty_hhincome).transform_filter(
        (alt.datum['Postal Code'] != 'US') & (alt.datum['County FIPS Code'] == 0)
    ).mark_rule(color='grey', opacity=0.5).encode(
        y='mean(Poverty Percent, All Ages)',
        strokeDash = alt.value([5,5])
    )
    ytext = yline.mark_text(align='left', dy=10, dx=100, opacity=0.5).encode(
        text=alt.value('National Average')
    )

    yline_s = alt.Chart(poverty_hhincome).transform_filter(
        (alt.datum['Postal Code'] != 'US') & (alt.datum['County FIPS Code'] == 0)
    ).mark_rule(color='black', size=2, opacity=0, strokeDash=[5, 5]).encode(
        y='Poverty Percent, All Ages',
        opacity=alt.condition(s2_1, alt.OpacityValue(1), alt.OpacityValue(0))
    )

    circle_cty = alt.Chart(poverty_hhincome, width=400, height=250).transform_filter(
        (alt.datum['Postal Code'] != 'US') & (alt.datum['County FIPS Code'] != 0) & (alt.datum['Poverty Percent, All Ages'] <= 40)
    ).mark_circle(size=20).encode(
        x=alt.X('Median Household Income:Q', scale=alt.Scale(domain=[20000,100000]), axis=alt.Axis(grid=False)),
        y=alt.Y('Poverty Percent, All Ages:Q', scale=alt.Scale(domain=[4,40]), title=None, axis=None),
        color=alt.condition(alt.datum.Name == 'Red Lake County', alt.value('red'), alt.value('blue')),
        tooltip=['Name', 'Median Household Income', alt.Tooltip('Poverty Percent, All Ages',  title='Poverty Percent, All Ages (%)')]
    ).add_selection(s2_2, s2_3)

    circle_rl = alt.Chart(poverty_hhincome, width=500, height=250).transform_filter(
        alt.datum.Name == 'Red Lake County'
    ).mark_circle(size=20, color='red').encode(
        x='Median Household Income:Q',
        y='Poverty Percent, All Ages:Q'
    )

    rxline = alt.Chart(poverty_hhincome, width=500).transform_filter(
        alt.datum['Name'] == 'Red Lake County'
    ).mark_rule(color='grey', opacity=0.5).encode(
        x='Median Household Income',
        strokeDash = alt.value([5,5])
    )

    rxtext = rxline.mark_text(align='left', y=20, dx=10).encode(
        text=alt.value('Red Lake')
    )

    ryline = alt.Chart(poverty_hhincome, width=500).transform_filter(
        alt.datum['Name'] == 'Red Lake County'
    ).mark_rule(color='grey', opacity=0.5).encode(
        y='Poverty Percent, All Ages',
        strokeDash = alt.value([5,5])
    )

    rytext = ryline.mark_text(align='left', dy=10, dx=100).encode(
        text=alt.value('Red Lake')
    )

    circle_cty = circle_cty.transform_filter(alt.datum['Postal Code'] == state).properties(
        title='Poverty Rate, Average House Income for ' + state
    )

    comp_2 = (circle + xline + xline_s + yline + yline_s + xtext + ytext) | (circle_cty + rxline + rxtext + ryline + rytext + circle_rl).resolve_scale(y='shared')

    return comp_2


# bar and dot plot for student per teacher and enrollment rate
def plot_v3(order_pt=False, order_er=False, state='MN', left=10, right=20, sort=False):
    sort_y = '-y' if sort else 'y'
    orderpt = 'ascending' if not order_pt else 'descending'
    orderer = 'ascending' if not order_er else 'descending'
    title_r = 'Bottom' if not order_er else 'Top'
    public_school_cty = public_school[public_school['STATE'] == state]

    bar_3 = alt.Chart(public_school_cty, width=400, height=200).mark_bar(width=15).transform_aggregate(
        groupby=['COUNTY'],
        ept_avg='mean(ENROLLMENT_PER_TEACHER)'
    ).encode(
        x=alt.X('COUNTY:N', sort=sort_y, title=None),
        y=alt.Y('ept_avg:Q', title='Average Enrollment per Teacher', scale=alt.Scale(domain=[0,22])),
        color=alt.Color('ept_avg:Q', scale=alt.Scale(scheme='greens', reverse=True), title=None, legend=None),
        tooltip=[alt.Tooltip('ept_avg:Q', title='Enrollment per Teacher')]
    ).transform_window(
        sort=[alt.SortField('ept_avg', order=orderpt)],
        ept_rank='rank(ept_avg)'
    ).transform_filter(
        (alt.datum.ept_rank <= left)
    ).properties(
        title='How many students is a teacher in charge of?'
    )

    dot_3 = alt.Chart(public_school_cty, width=400, height=200).mark_point(filled=True).transform_aggregate(
        groupby=['COUNTY'],
        er_avg='mean(ENROLLMENT_RATE)',
        ept_avg='mean(ENROLLMENT_PER_TEACHER)'
    ).encode(
        x=alt.X('COUNTY:N', sort=sort_y, title=None),
        y=alt.Y('er_avg:Q', title='Average Enrollment Rate', scale=alt.Scale(domain=[85,100])),
        size=alt.Size('ept_avg:Q', title='Average Enrollment per Teacher'),
        color=alt.Color('er_avg:Q', scale=alt.Scale(scheme='reds', reverse=True), title=None, legend=None),
        tooltip=[alt.Tooltip('er_avg:Q', title='Average Enrollment Rate (%)')]
    ).transform_window(
        sort=[alt.SortField('er_avg', order=orderer)],
        er_rank='rank(er_avg)'
    ).transform_filter(
        (alt.datum.er_rank <= right)
    ).properties(
        title=title_r + ' ' + str(right) + ' counties of enrollment rate in ' + state + ' (%)'
    )

    lline = alt.Chart(public_school[public_school['COUNTY'] == 'RED LAKE']).mark_rule().encode(
        y='mean(ENROLLMENT_PER_TEACHER)',
        strokeDash = alt.value([5,5])
    )

    lt = lline.mark_text(dx=-170, dy=-10).encode(
        text=alt.value('Red Lake')
    )

    rline = alt.Chart(public_school[public_school['COUNTY'] == 'RED LAKE']).mark_rule().encode(
        y='mean(ENROLLMENT_RATE)',
        strokeDash = alt.value([5,5])
    )

    rt = rline.mark_text(dx=-170, dy=-10).encode(
        text=alt.value('Red Lake')
    )

    comp_3 = ((bar_3 + lline + lt) | (dot_3 + rline + rt)).resolve_scale(color='independent')

    return comp_3


# heatmap, line and scatter plot of top five mortal diseases
def plot_v4(disease_list, log=False, line=False, year_l=1980, year_r=2014):
    log_y = alt.Y('MRP100k_log:Q', title='log(Death per 100000 people)') if log else alt.Y('MRP100k:Q', title='Death per 100000 people')
    share_y = alt.Y('yy:N', title=None, axis=None) if not line else alt.Y('yy:N', title=None)
    share_y2 = alt.Y('yy2:N', title=None, axis=None) if not line else alt.Y('yy2:N', title=None)

    idx_lb, idx_ub = year_list.index(str(year_l)), year_list.index(str(year_r))
    year_interval = year_list[idx_lb:(idx_ub + 1)]

    disease_list = [add_wrapping(d) for d in disease_list]

    s4_1 = alt.selection_single(on='click', empty='none')
    s4_2 = alt.selection_single(on='mouseover', empty='none')

    heatmap = alt.Chart(mortality_mn_top5, width=350, height=250).mark_rect(opacity=0.8).transform_calculate(
        yy="split(datum.Category, '+')"
    ).encode(
        x=alt.X('Year:N', title=None),
        y=alt.Y('yy:N', title=None),
        color=alt.Color('MRP100k:Q', scale=alt.Scale(scheme='reds'), title='Death per 100000 people')
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Year', oneOf=year_interval)
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Category', oneOf=disease_list)
    )

    text = heatmap.mark_text().encode(
        text=alt.Text('MRP100k:Q'),
        color=alt.value('blue')
    )

    highlight = alt.Chart(mortality_mn_top5, width=350, height=250).mark_rect(filled=False, color='red').transform_calculate(
        yy3="split(datum.Category, '+')"
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Year', oneOf=[year_interval[0], year_interval[-1]])
    ).encode(
        x=alt.X('Year:N', title=None),
        y=alt.Y('yy3:N', title=None),
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Category', oneOf=disease_list)
    )

    lines_4 = alt.Chart(mortality_mn_top5, width=350, height=250).mark_line(size=3).transform_calculate(
        yy="split(datum.Category, '+')"
    ).encode(
        x=alt.X('Year:N', title=None),
        y=log_y,
        color=alt.Color('yy:N', title='Category'),
        tooltip=[alt.Tooltip('MRP100k', title='Mortality per 100k people')]
    ).properties(
        title='Mortality of Top 5 Diseases in Red Lake County'
    ).add_selection(s4_1).encode(
        strokeDash=alt.condition(s4_1, alt.value([0]), alt.value([8,2])),
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Year', oneOf=year_interval)
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Category', oneOf=disease_list)
    )

    liner1 = alt.Chart(mortality_top5_plot, width=350, height=250).mark_circle(color='red').transform_calculate(
        yy="split(datum.Category, '+')"
    ).encode(
        x='% Change in Mortality Rate, 1980-2014',
        y=share_y,
        tooltip=['% Change in Mortality Rate, 1980-2014']
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Category', oneOf=disease_list)
    )

    tl1 = liner1.transform_filter(
        alt.datum.Category == 'Chronic+respiratory diseases'
    ).mark_text(align='left', dy=-10, dx=-20).encode(
        text=alt.value('National')
    )

    tl2 = liner1.transform_filter(
        alt.datum.Category == 'Chronic+respiratory diseases'
    ).mark_text(align='right', dy=10, dx=-30).encode(
        text=alt.value('Red Lake')
    )

    liner2 = alt.Chart(mortality_mn_plot).mark_circle(color='blue').transform_calculate(
        yy2="split(datum.Category, '+')"
    ).encode(
        x='% Change in Mortality Rate, 1980-2014',
        y=share_y2,
        tooltip=['% Change in Mortality Rate, 1980-2014']
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Category', oneOf=disease_list)
    )

    
    if line:
        comp_4 = lines_4 | ((liner1 + liner2) + tl1 + tl2)
    else:
        comp_4 = (heatmap + text + highlight) | ((liner1 + liner2).resolve_scale(y='shared') + tl1 + tl2)

    return comp_4



# display contents

st.title('Does Red Lake County deserve "the WORST place to live in America"?')

st.header('Introduction')

st.markdown('According to USDA, Minnesota has the lowest average natural amenity rating across the country as the graph below shows, and Red Lake County ranks last in Minnesota. Christopher Ingraham claims in his article that Red Lake County is "the worst place to live in America" based on the natural amenity statistics.')

st.image('NA.png', width=1000)

st.markdown('In the following interactive visualization, we are making a doubt on the claim of Christopher by integrating data from various sources to reflect on the true facts of Red Lake County.')

st.header('Charts')

st.markdown('### *Unemployment Rate, greatly BELOW average.*')

st.markdown('The unemployment rate of Red Lake County is well below the country average, the place is definitely not the “worst” in terms of job market.')

st.altair_chart(plot_v1())

st.markdown('### *Low Poverty Rate, but Median Household Income is a bit Lagging Behind.*')

st.markdown('The poverty rate of Red Lake County is about 4% lower than national average. The median household income is a little bit lower than country average. Since Minnesota is far from any prosperous areas in the US, where income is great and the living cost is also high. Red Lake is lagging behind mainly for geographical reasons.')

states_list = ['MN'] + [s for s in list(poverty_hhincome['Postal Code'].unique()) if s not in ['MN', 'US']]
state1 = st.selectbox('Choose a State to view on the RIGHT graph:', states_list, key='0')
st.altair_chart(plot_v2(state=state1))

st.markdown('### *Educational Resource: Abundant, but need more participation.*')

st.markdown('* Each public school teacher in Red Lake is in charge of about 11 students, one person below national average, which means Red Lake is not in short of faculty members.')

st.markdown('* The average of enrollment rate in Red Lake is about 92%, about 1.5% higher than national average, but still ranks among 20 lowest within the state.')

state2 = st.selectbox('Choose a State to view on BOTH graph:', states_list, key='1')
tt = st.checkbox('Bottom Values for Students per Teacher')
ter = st.checkbox('Top Values for Enrollment Rate')
sort_y = st.checkbox('Reverse Order')
slider_l = st.slider('Select Number of Data to Display on the LEFT Plot', 10, 20, 15)
slider_r = st.slider('Select Number of Data to Display on the RIGHT Plot', 10, 20, 20)
st.altair_chart(plot_v3(order_er=ter, order_pt=tt, state=state2, sort=sort_y, left=slider_l, right=slider_r))


st.markdown('### *Expanded Research: Mortality & Disease*')

st.markdown('* The top five diseases causing death in Red Lake County have a mortality from 10  up to 444 per 100000 people during 1980 and 2014. The most fatal disease is cardiovascular diseases, which deals with heart and blood vessels.')

st.markdown('* Cardiovascular dramatically drop by over 40% during the past 35 years, while diabetes, urogenital, blood and endocrine diseases increse by nearly 80%.')

st.markdown('* Among top 5 diseases, Red Lake does a good job in 4 of them compared to national data in terms of change, diabetes remains a great challenge to its residents.')

radio = st.radio('Select Plot Type', ('Line', 'Heatmap'))
slider_year_l, slider_year_r = st.slider('Choose Year Range:', 1980, 2014, (1980, 2014), step=5)
if slider_year_r == 2015:
    slider_year_r -= 1

diseases = ['Cardiovascular diseases', 'Chronic respiratory diseases',
'Cirrhosis and other chronic liver diseases', 'Diabetes, urogenital, blood, and endocrine diseases',
'Diarrhea, lower respiratory, and other common infectious diseases']

multi = st.multiselect('Select Disease:', diseases, default=diseases)

if radio == 'Line':
    log = st.checkbox('Use Log Y Scale')
    st.altair_chart(plot_v4(log=log, line=True, year_l=slider_year_l, year_r=slider_year_r, disease_list=multi))
else:
    st.altair_chart(plot_v4(year_l=slider_year_l, year_r=slider_year_r, disease_list=multi))

st.header('Conclusion')

st.markdown('Red Lake is misunderstood by the one-sided metric of natural amenity. Despite the terrible natural environment, Red Lake is a fair place with low unemployment rate, low poverty rate, relatively good education resource, as well as a good control of mortality.')
