import os

# irrigated gages as of 7/22/2022
TARGET_IRRIGATED_GAGES = ['06016000',
                          '06018500',
                          '06019500',
                          '06025500',
                          '06033000',
                          '06035000',
                          '06036650',
                          '06041000',
                          '06052500',
                          '06054500',
                          '06066500',
                          '06078200',
                          '06089000',
                          '06090800',
                          '06099000',
                          '06101500',
                          '06108000',
                          '06109500',
                          '06120500',
                          '06126500',
                          '06127500',
                          '06130500',
                          '06139500',
                          '06142400',
                          '06166000',
                          '06177000',
                          '06177500',
                          '06185500',
                          '06192500',
                          '06195600',
                          '06200000',
                          '06205000',
                          '06207500',
                          '06208500',
                          '06211000',
                          '06211500',
                          '06214500',
                          '06218500',
                          '06227600',
                          '06228000',
                          '06235500',
                          '06236100',
                          '06274300',
                          '06279500',
                          '06280300',
                          '06281000',
                          '06285100',
                          '06287000',
                          '06289600',
                          '06289820',
                          '06294000',
                          '06295000',
                          '06306300',
                          '06307616',
                          '06308500',
                          '06309000',
                          '06317000',
                          '06324500',
                          '06324970',
                          '06326500',
                          '06329500',
                          '09019500',
                          '09034250',
                          '09038500',
                          '09057500',
                          '09058000',
                          '09067000',
                          '09070000',
                          '09070500',
                          '09085000',
                          '09085100',
                          '09095500',
                          '09105000',
                          '09112500',
                          '09114500',
                          '09115500',
                          '09119000',
                          '09132500',
                          '09144250',
                          '09146200',
                          '09147000',
                          '09147025',
                          '09147500',
                          '09149500',
                          '09152500',
                          '09163500',
                          '09166500',
                          '09169500',
                          '09171100',
                          '09177000',
                          '09180000',
                          '09180500',
                          '09188500',
                          '09205000',
                          '09209400',
                          '09210500',
                          '09224700',
                          '09229500',
                          '09237450',
                          '09237500',
                          '09239500',
                          '09251000',
                          '09253000',
                          '09255000',
                          '09257000',
                          '09260000',
                          '09260050',
                          '09277500',
                          '09288180',
                          '09295000',
                          '09302000',
                          '09304200',
                          '09304500',
                          '09304800',
                          '09306200',
                          '09306222',
                          '09306290',
                          '09306500',
                          '09314500',
                          '09315000',
                          '09328500',
                          '09337500',
                          '09342500',
                          '09346400',
                          '09349800',
                          '09354500',
                          '09355500',
                          '09361500',
                          '09363500',
                          '09364500',
                          '09365000',
                          '09367500',
                          '09368000',
                          '09371000',
                          '09371010',
                          '09379500',
                          '09382000',
                          '09386900',
                          '09390500',
                          '09403600',
                          '09404450',
                          '09406000',
                          '09408135',
                          '09408150',
                          '09409100',
                          '09410100',
                          '09413000',
                          '09413200',
                          '09413500',
                          '09415000',
                          '09418500',
                          '09418700',
                          '09419000',
                          '09419665',
                          '09419700',
                          '09419753',
                          '09419800',
                          '09448500',
                          '09466500',
                          '09482000',
                          '09486500',
                          '09486520',
                          '09504420',
                          '09504500',
                          '09505350',
                          '12323600',
                          '12323770',
                          '12324200',
                          '12331500',
                          '12334510',
                          '12334550',
                          '12335500',
                          '12340000',
                          '12340500',
                          '12344000',
                          '12350250',
                          '12352500',
                          '12353000',
                          '12354500',
                          '12366000',
                          '12372000',
                          '12388700',
                          '12389000',
                          '12395500',
                          '12396500',
                          '12409000',
                          '12422500',
                          '12424000',
                          '12433000',
                          '12433200',
                          '12448998',
                          '12452500',
                          '12459000',
                          '12462500',
                          '12465000',
                          '12484500',
                          '12500450',
                          '12502500',
                          '12508990',
                          '12510500',
                          '13011900',
                          '13018350',
                          '13018750',
                          '13022500',
                          '13027500',
                          '13032500',
                          '13037500',
                          '13038500',
                          '13046000',
                          '13049500',
                          '13050500',
                          '13052200',
                          '13055000',
                          '13056500',
                          '13057940',
                          '13062500',
                          '13066000',
                          '13069500',
                          '13073000',
                          '13075000',
                          '13075500',
                          '13077000',
                          '13078000',
                          '13082500',
                          '13090000',
                          '13094000',
                          '13105000',
                          '13112000',
                          '13116500',
                          '13127000',
                          '13132500',
                          '13141500',
                          '13142500',
                          '13147900',
                          '13148500',
                          '13168500',
                          '13172500',
                          '13181000',
                          '13213000',
                          '13213100',
                          '13245000',
                          '13246000',
                          '13247500',
                          '13249500',
                          '13250000',
                          '13251000',
                          '13258500',
                          '13269000',
                          '13302500',
                          '13305000',
                          '13316500',
                          '13317000',
                          '13333000',
                          '13342450',
                          '13344500',
                          '13346800',
                          '13351000',
                          '14015000',
                          '14018500',
                          '14033500',
                          '14034470',
                          '14034480',
                          '14034500',
                          '14038530',
                          '14046000',
                          '14046500',
                          '14048000',
                          '14076500',
                          '14087400',
                          '14092500',
                          '14103000',
                          '14113000',
                          '14120000',
                          '14123500',
                          '14137000',
                          '14141500',
                          '14142500',
                          '14152000',
                          '14153500',
                          '14155500',
                          '14157500',
                          '14163900',
                          '14165000',
                          '14166000',
                          '14166500',
                          '14169000',
                          '14170000',
                          '14174000',
                          '14183000',
                          '14187500',
                          '14188800',
                          '14189000',
                          '14190500',
                          '14191000',
                          '14202980',
                          '14203500',
                          '14207500',
                          '14211500',
                          '14211550',
                          '14211720',
                          '14243000',
                          '06024450',
                          '06061500',
                          '06071300',
                          '06076690',
                          '09041090',
                          '09129600',
                          '09143500',
                          '09242500',
                          '09330000',
                          '09333500',
                          '09442000',
                          '12324590',
                          '12448500',
                          '12449500',
                          '12449950',
                          '13058000',
                          '13075910',
                          '14101500',
                          '14184100',
                          '06295113',
                          '14152500',
                          '06220800',
                          '06305700',
                          '09416000',
                          '14147500',
                          '13266000']

# irrigated gages hydrographs verified as of 7/22/2022
VERIFIED_IRRIGATED_HYDROGRAPHS = ['06025500', '06033000', '06052500', '06054500', '06066500', '06089000', '06090800',
                                  '06101500',
                                  '06108000', '06109500', '06126500', '06130500', '06177000', '06185500', '06192500',
                                  '06195600',
                                  '06200000', '06207500', '06214500', '06228000', '06235500', '06274300', '06279500',
                                  '06280300',
                                  '06281000', '06285100', '06287000', '06294000', '06295000', '06306300', '06307616',
                                  '06308500',
                                  '06317000', '06324500', '06324970', '06326500', '06329500', '09034250', '09038500',
                                  '09057500',
                                  '09058000', '09070000', '09070500', '09085000', '09085100', '09095500', '09105000',
                                  '09112500',
                                  '09114500', '09119000', '09132500', '09144250', '09146200', '09147000', '09147500',
                                  '09149500',
                                  '09152500', '09163500', '09166500', '09169500', '09171100', '09180000', '09180500',
                                  '09205000',
                                  '09209400', '09210500', '09224700', '09237500', '09239500', '09251000', '09255000',
                                  '09260000',
                                  '09277500', '09288180', '09295000', '09302000', '09304200', '09304500', '09304800',
                                  '09306222',
                                  '09306290', '09306500', '09315000', '09328500', '09330000', '09337500', '09342500',
                                  '09346400',
                                  '09349800', '09354500', '09355500', '09361500', '09364500', '09365000', '09368000',
                                  '09371000',
                                  '09379500', '09382000', '09386900', '09390500', '09403600', '09404450', '09406000',
                                  '09413000',
                                  '09413200', '09415000', '09416000', '09419000', '09448500', '09466500', '09504420',
                                  '09504500',
                                  '09505350', '12323770', '12324200', '12334510', '12334550', '12340000', '12340500',
                                  '12344000',
                                  '12353000', '12354500', '12372000', '12388700', '12389000', '12395500', '12396500',
                                  '12409000',
                                  '12422500', '12424000', '12433000', '12449950', '12452500', '12459000', '12462500',
                                  '12465000',
                                  '12484500', '12500450', '12502500', '13018750', '13022500', '13027500', '13032500',
                                  '13037500',
                                  '13038500', '13046000', '13049500', '13050500', '13052200', '13055000', '13056500',
                                  '13057940',
                                  '13058000', '13062500', '13066000', '13069500', '13073000', '13075000', '13075500',
                                  '13077000',
                                  '13078000', '13082500', '13090000', '13094000', '13105000', '13116500', '13127000',
                                  '13132500',
                                  '13141500', '13142500', '13147900', '13148500', '13168500', '13172500', '13181000',
                                  '13245000',
                                  '13246000', '13247500', '13249500', '13251000', '13258500', '13266000', '13269000',
                                  '13302500',
                                  '13305000', '13316500', '13317000', '13333000', '13346800', '13351000', '14015000',
                                  '14018500',
                                  '14033500', '14034470', '14034500', '14046000', '14046500', '14076500', '14087400',
                                  '14092500',
                                  '14103000', '14113000', '14120000', '14123500', '14137000', '14141500', '14142500',
                                  '14152000',
                                  '14153500', '14155500', '14157500', '14166000', '14166500', '14169000', '14170000',
                                  '14174000',
                                  '14183000', '14189000', '14190500', '14191000', '14203500', '14207500', '14211500',
                                  '14211720']

# manually inspected gages to exclude
EXCLUDE_STATIONS = ['05015500', '06154400', '06311000', '06329590', '06329610', '06329620',
                    '09125800', '09131495', '09147022', '09213700', '09362800', '09398300',
                    '09469000', '09509501', '09509502', '12371550', '12415500', '12452000',
                    '13039000', '13106500', '13115000', '13119000', '13126000', '13142000',
                    '13148200', '13171500', '13174000', '13201500', '13238500', '13340950',
                    '14149000', '14150900', '14153000', '14155000', '14162100', '14168000',
                    '14180500', '14186100', '14186600', '14207740', '14207770', '14234800',
                    '12472600', '06020600', '06088500', '06253000', '12472600', '12513000',
                    '12324680', '12329500', '12467000', '13108150', '13153500', '13152500',
                    '09211200', '09261000', '09128000', '09519800', '13153500', '09372000',
                    '09371492', '12398600', '09520500', '09489000', '09519800', '09520500',
                    '09519800', '09520500', '09469500', '09474000', '06185110', '13183000',
                    '13171620', '13135000', '09106150', '06307500', '14238000', '13081500',
                    '12465400', '09386900']

# lowest gages on the major systems
SYSTEM_STATIONS = ['06109500', '06329500', '09180500', '09315000',
                   '09379500', '12396500', '13269000', '13317000']
# gages of arbitrary interest
SELECTED_SYSTEMS = ['06109500', '06329500', '09180500', '09315000',
                    '09379500', '09466500', '12389000', '12510500',
                    '13269000', '13317000', '14048000',
                    '14103000', '14211720']

STATION_BASINS = {'06025500': 'missouri',
                  '06033000': 'missouri',
                  '06052500': 'missouri',
                  '06054500': 'missouri',
                  '06066500': 'missouri',
                  '06089000': 'missouri',
                  '06090800': 'missouri',
                  '06101500': 'missouri',
                  '06108000': 'missouri',
                  '06109500': 'missouri',
                  '06126500': 'missouri',
                  '06130500': 'missouri',
                  '06177000': 'missouri',
                  '06185500': 'missouri',
                  '06192500': 'missouri',
                  '06195600': 'missouri',
                  '06200000': 'missouri',
                  '06207500': 'missouri',
                  '06214500': 'missouri',
                  '06228000': 'missouri',
                  '06235500': 'missouri',
                  '06274300': 'missouri',
                  '06279500': 'missouri',
                  '06280300': 'missouri',
                  '06281000': 'missouri',
                  '06285100': 'missouri',
                  '06287000': 'missouri',
                  '06294000': 'missouri',
                  '06295000': 'missouri',
                  '06306300': 'missouri',
                  '06307616': 'missouri',
                  '06308500': 'missouri',
                  '06317000': 'missouri',
                  '06324500': 'missouri',
                  '06324970': 'missouri',
                  '06326500': 'missouri',
                  '06329500': 'missouri',
                  '09034250': 'colorado',
                  '09038500': 'colorado',
                  '09057500': 'colorado',
                  '09058000': 'colorado',
                  '09070000': 'colorado',
                  '09070500': 'colorado',
                  '09085000': 'colorado',
                  '09085100': 'colorado',
                  '09095500': 'colorado',
                  '09105000': 'colorado',
                  '09112500': 'colorado',
                  '09114500': 'colorado',
                  '09119000': 'colorado',
                  '09132500': 'colorado',
                  '09144250': 'colorado',
                  '09146200': 'colorado',
                  '09147000': 'colorado',
                  '09147500': 'colorado',
                  '09149500': 'colorado',
                  '09152500': 'colorado',
                  '09163500': 'colorado',
                  '09166500': 'colorado',
                  '09169500': 'colorado',
                  '09171100': 'colorado',
                  '09180000': 'colorado',
                  '09180500': 'colorado',
                  '09205000': 'colorado',
                  '09209400': 'colorado',
                  '09210500': 'colorado',
                  '09224700': 'colorado',
                  '09237500': 'colorado',
                  '09239500': 'colorado',
                  '09251000': 'colorado',
                  '09255000': 'colorado',
                  '09260000': 'colorado',
                  '09277500': 'colorado',
                  '09288180': 'colorado',
                  '09295000': 'colorado',
                  '09302000': 'colorado',
                  '09304200': 'colorado',
                  '09304500': 'colorado',
                  '09304800': 'colorado',
                  '09306222': 'colorado',
                  '09306290': 'colorado',
                  '09306500': 'colorado',
                  '09315000': 'colorado',
                  '09328500': 'colorado',
                  '09330000': 'colorado',
                  '09337500': 'colorado',
                  '09342500': 'colorado',
                  '09346400': 'colorado',
                  '09349800': 'colorado',
                  '09354500': 'colorado',
                  '09355500': 'colorado',
                  '09361500': 'colorado',
                  '09364500': 'colorado',
                  '09365000': 'colorado',
                  '09368000': 'colorado',
                  '09371000': 'colorado',
                  '09379500': 'colorado',
                  '09382000': 'colorado',
                  '09386900': 'colorado',
                  '09390500': 'colorado',
                  '09403600': 'colorado',
                  '09404450': 'colorado',
                  '09406000': 'colorado',
                  '09413000': 'colorado',
                  '09413200': 'colorado',
                  '09415000': 'colorado',
                  '09416000': 'colorado',
                  '09419000': 'colorado',
                  '09448500': 'colorado',
                  '09466500': 'colorado',
                  '09504420': 'colorado',
                  '09504500': 'colorado',
                  '09505350': 'colorado',
                  '12323770': 'columbia',
                  '12324200': 'columbia',
                  '12334510': 'columbia',
                  '12334550': 'columbia',
                  '12340000': 'columbia',
                  '12340500': 'columbia',
                  '12344000': 'columbia',
                  '12353000': 'columbia',
                  '12354500': 'columbia',
                  '12372000': 'columbia',
                  '12388700': 'columbia',
                  '12389000': 'columbia',
                  '12395500': 'columbia',
                  '12396500': 'columbia',
                  '12409000': 'columbia',
                  '12422500': 'columbia',
                  '12424000': 'columbia',
                  '12433000': 'columbia',
                  '12449950': 'columbia',
                  '12452500': 'columbia',
                  '12459000': 'columbia',
                  '12462500': 'columbia',
                  '12465000': 'columbia',
                  '12484500': 'columbia',
                  '12500450': 'columbia',
                  '12502500': 'columbia',
                  '13018750': 'columbia',
                  '13022500': 'columbia',
                  '13027500': 'columbia',
                  '13032500': 'columbia',
                  '13037500': 'columbia',
                  '13038500': 'columbia',
                  '13046000': 'columbia',
                  '13049500': 'columbia',
                  '13050500': 'columbia',
                  '13052200': 'columbia',
                  '13055000': 'columbia',
                  '13056500': 'columbia',
                  '13057940': 'columbia',
                  '13058000': 'columbia',
                  '13062500': 'columbia',
                  '13066000': 'columbia',
                  '13069500': 'columbia',
                  '13073000': 'columbia',
                  '13075000': 'columbia',
                  '13075500': 'columbia',
                  '13077000': 'columbia',
                  '13078000': 'columbia',
                  '13082500': 'columbia',
                  '13090000': 'columbia',
                  '13094000': 'columbia',
                  '13105000': 'columbia',
                  '13116500': 'columbia',
                  '13127000': 'columbia',
                  '13132500': 'columbia',
                  '13141500': 'columbia',
                  '13142500': 'columbia',
                  '13147900': 'columbia',
                  '13148500': 'columbia',
                  '13168500': 'columbia',
                  '13172500': 'columbia',
                  '13181000': 'columbia',
                  '13245000': 'columbia',
                  '13246000': 'columbia',
                  '13247500': 'columbia',
                  '13249500': 'columbia',
                  '13251000': 'columbia',
                  '13258500': 'columbia',
                  '13266000': 'columbia',
                  '13269000': 'columbia',
                  '13302500': 'columbia',
                  '13305000': 'columbia',
                  '13316500': 'columbia',
                  '13317000': 'columbia',
                  '13333000': 'columbia',
                  '13346800': 'columbia',
                  '13351000': 'columbia',
                  '14015000': 'columbia',
                  '14018500': 'columbia',
                  '14033500': 'columbia',
                  '14034470': 'columbia',
                  '14034500': 'columbia',
                  '14046000': 'columbia',
                  '14046500': 'columbia',
                  '14076500': 'columbia',
                  '14087400': 'columbia',
                  '14092500': 'columbia',
                  '14103000': 'columbia',
                  '14113000': 'columbia',
                  '14120000': 'columbia',
                  '14123500': 'columbia',
                  '14137000': 'columbia',
                  '14141500': 'columbia',
                  '14142500': 'columbia',
                  '14152000': 'columbia',
                  '14153500': 'columbia',
                  '14155500': 'columbia',
                  '14157500': 'columbia',
                  '14166000': 'columbia',
                  '14166500': 'columbia',
                  '14169000': 'columbia',
                  '14170000': 'columbia',
                  '14174000': 'columbia',
                  '14183000': 'columbia',
                  '14189000': 'columbia',
                  '14190500': 'columbia',
                  '14191000': 'columbia',
                  '14203500': 'columbia',
                  '14207500': 'columbia',
                  '14211500': 'columbia',
                  '14211720': 'columbia'}

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================