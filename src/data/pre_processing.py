import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

class PreProcess:

    def __init__(self, data, X):
        self.data = data
        self.X = X
        self.sc = StandardScaler()

    def scale(self):
        self.X_scaled = self.sc.fit_transform(self.X)
        return pd.DataFrame(self.X_scaled, columns=['NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash', 'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore', 'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash', 'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress', 'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname', 'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath', 'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks', 'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms', 'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction', 'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch', 'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow', 'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle', 'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT', 'PctExtResourceUrlsRT', 'AbnormalExtFormActionR', 'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'])

    def check_null(self):
        # print the count to null values
        print('Before imputation:')
        print(self.data.isnull().sum()) 

        # handle the null value by imputation
        self.data.fillna(0)
        print('------------') 

        # print the count if all null values are handled
        print('After imputation:')
        print(self.data.isnull().sum()) 

class LoadData():

    def __init__(self, URL):
        self.URL = URL

    def readcsv(self):
        return pd.read_csv(self.URL)



    