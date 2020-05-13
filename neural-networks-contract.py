# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time
stemmer = LancasterStemmer()

training_data = []
training_data.append({"class":"parties", "sentence":"THIS LOAN AGREEMENT, dated effective as of September 20, 2018, is entered into between EMPIRE LOUISIANA, LLC, a Delaware limited liability company, (Borrower), and CROSSFIRST BANK, a Kansas banking corporation (Bank)"})
training_data.append({"class":"condition", "sentence":"Borrower has requested Bank to establish a senior revolving line of credit facility in favor of Borrower in the maximum principal amount of FIVE MILLION and NO/100 DOLLARS ($5,000,000.00), subject to the Revolver Commitment Amount (initially stipulated to be $1,350,000.00) and the Collateral Borrowing Base limitations hereof, until the Revolver Final Maturity Date, to be evidenced by Borrower's Promissory Note payable to the order of Bank and dated as of even date herewith in the stated face principal amount of $5,000,000.00 (as renewed, extended, rearranged, substituted, replaced, amended or otherwise modified from time to time, collectively the Note), the proceeds of which may be used for the acquisition of oil and natural gas reserves, the development of oil and natural gas reserves, the general working capital and capital expenditures of the Borrower, the issuance of standby letters of credit"})
training_data.append({"class":"condition", "sentence":"Bank is willing to establish the Revolver Commitment and make the Revolver Loan advances from time to time hereunder to Borrower in the maximum principal amount of $5,000,000.00, subject to the Revolver Commitment Amount (initially stipulated to be $1,350,000.00) and the Collateral Borrowing Base, all upon the terms and conditions herein set forth, and upon Borrower's granting in favor of Bank a continuing and continuous, first priority mortgage lien, pledge of and security interest in not less than 80% of Borrower's producing oil, gas and other leasehold and mineral interests in counties in the State of Louisiana, along with certain contract rights, all as more particularly described and defined in the Security Instruments (as hereinafter defined), as collateral and security for all Indebtedness"})
training_data.append({"class":"Commitment", "sentence":"Bank agrees, upon the terms and subject to the conditions hereinafter set forth, to make revolving loan advances to Borrower from the Closing Date until the Revolver Final Maturity Date, or until such later date as Bank shall have extended its Revolver Commitment in writing unless the Revolver Commitment shall be sooner terminated pursuant to the provisions of this Agreement"})
training_data.append({"class":"Note", "sentence":"On the Closing Date, Borrower shall execute and deliver to the order of Bank its promissory note instrument in the stated face principal amount of $5,000,000.00.  The Revolver Note shall be dated as of the Closing Date and shall bear interest on unpaid balances of principal from time to time outstanding at a variable annual rate equal from day to day to the Base Rate plus one hundred fifty basis points (1.50%).  The Revolver Note shall be payable as set forth therein.  After maturity (whether by acceleration or otherwise), the Revolver Note shall bear interest at a per annum rate equal from day to day to the Default Rate payable on demand, unless there has been no default in Borrower's payment obligations (other than Borrower's failure to pay all unpaid principal and all accrued but unpaid interest due and payable at the Revolver Final Maturity Date) and Borrower and Bank are negotiating a renewal or extension of the Revolver Note, in which circumstance the non-Default Rate specified herein shall continue to apply, but only until Bank deems negotiations complete, in its sole discretion.  Interest shall be calculated on the basis of a year of 360 days, but assessed for the actual number of days elapsed in each accrual period"})
training_data.append({"class":"Sale of Mortgaged Property", "sentence":"In the event any undivided interest in any of the Mortgaged Property is sold and causes a Deficiency (as defined in Section 4.3 hereof), the sales proceeds of any such sale shall be applied initially to the outstanding principal balance of the Revolver Note, then to accrued interest under the Revolver Note; provided, however, no such sale shall occur except as permitted in Section 6.16 hereof or in the Mortgages or without the prior written consent of Bank, not to be unreasonably withheld, conditioned or delayed"})
training_data.append({"class":"Loan Origination Fee", "sentence":"Borrower shall pay to Bank a fully earned and non-refundable loan origination fee equal to $13,500.00 (one hundred basis points (1.00%) on the initial $1,350,000.00 Revolver Commitment Amount).  If and to the extent the Revolver Commitment Amount is increased above the current stipulated amount thereof $1,350,000.00 from time to time, an additional fully earned and non-refundable loan amendment fee of twenty five basis points (0.25%) shall be due and payable to Bank on the amount of such increase concurrent with the effectiveness of any such increase(s)"})
training_data.append({"class":"Non-usage Fee", "sentence":"From and following the Closing Date to the date the Revolver Commitment expires or is otherwise terminated, Borrower shall pay Bank a per annum fee in anamount equal to the Revolver Commitment Amount less (a) the actual daily balance of the sum of the Revolver Loans outstanding, multiplied by (b) twenty five basis points 0.25% computed on the basis of a calendar year of 360 days but assessed for the actual number of days elapsed during each quarterly accrual period.  Such Revolver non-usage fee is to be paid quarterly in arrears, commencing with the calendar quarter ending December 31, 2018, and payable within ten (10) days following Borrower's receipt of a written invoice therefor reasonably detailing Bank's calculation thereof"})
training_data.append({"class":"Letters of Credit", "sentence":"Upon Borrower's application from time to time by use of Bank's standard form Letter of Credit Application Agreement and subject to the terms and provisions therein and herein set forth, Bank agrees to issue standby letters of credit on behalf of Borrower under the Revolver Commitment in an aggregate face amount not exceeding fifteen percent 15% of the lesser of the Collateral Borrowing Base or the Revolver Commitment Amount then in effect, provided that (i) no letters of credit will be issued on behalf of or on the account of Borrower, except only for letters of credit with maturities not exceeding one year that contain automatic renewal language approved by Bank, and (ii) no letter of credit will be issued on behalf of or for the account of Borrower"})
training_data.append({"class":"Termination of Agreement", "sentence":"If and to the extent any Hedge Agreement or similar price protection or derivative product (interest rate or commodity risk management device, protection agreement or otherwise) of Borrower is used in calculation of the Collateral Borrowing Base, any such Hedge Agreement issued cannot be cancelled, liquidated or unwound thereby without the prior written consent of Bank"})
training_data.append({"class":"Collateral Borrowing Base", "sentence":"Borrower will not knowingly request, nor will it knowingly accept, the proceeds of any Revolver Loan or advance under the Revolver Note at any time when the amount thereof, together with the sum of the outstanding and unpaid principal amount of the Revolver Note plus the Letter of Credit Exposure exceeds the lesser of (i) Collateral Borrowing Base or (ii) the then applicable Revolver Commitment Amount"})
training_data.append({"class":"Variance from Collateral Borrowing Base", "sentence":"Any Revolver Loan advance shall be conclusively presumed to have been made to Borrower by Bank under the terms and provisions hereof and shall be secured by all of the Collateral and security described or referred to herein or in the Mortgages, whether or not such loan conforms in all respects to the terms and provisions hereof.  If Bank should (for the convenience of Borrower or for any other reason) make loans or advances which would cause the unpaid principal amount of the Revolver Note plus outstanding and unfunded Letters of Credit to exceed the amount of the applicable Collateral Borrowing Base, no such variance, change or departure shall prevent any such loan or loans from being secured by the Collateral and the security created or intended to be created herein or in the Security Instruments"})
training_data.append({"class":"Late Fee", "sentence":"Any principal or interest due under this Agreement, the Revolver Note, or any other Loan Document which is not paid within 10 days after its due date (whether as stated, by acceleration or otherwise) shall be subject to a late payment charge of five percent (5.00%) of the total payment due, in addition to the payment of interest.  Borrower agrees to pay and stipulates that five percent (5.00%) of the total payment due in a reasonable amount for a late payment charge.  Borrower shall pay the late payment charge upon demand by Bank or, if billed, within the time specified, and in immediately available funds, US Dollars"})
training_data.append({"class":"ACH Debits", "sentence":"To effectuate any payment due under the Agreement, the Revolver Note or any other Loan Document, Borrower hereby authorizes Bank to initiate debit entries to its operating account at Bank and to debit the same to such account.  This authorization to initiate debit entries shall remain in full force and effect until Bank has received written notification of its termination in such time and in such manner as to afford Bank a reasonable opportunity to act on it"})
training_data.append({"class":"Payment of Fees", "sentence":"All fees payable under Sections 2.4, 2.5, 2.6 and 2.10 above  shall be paid on the dates due, in immediately available funds, US Dollars, to Bank and shall be fully earned and non refundable under any circumstances"})
training_data.append({"class":"MCR", "sentence":"As of the first day of each calendar month, commencing November 10, 2018, the Revolver Commitment Amount shall be automatically reduced by the monthly Commitment reduction amount, which initially is stipulated to be $10,000.00 per month. Commencing on December 31, 2018, and from time to time thereafter, the MCR will be subject to adjustment by the Bank in its discretion at each semi-annual Collateral Borrowing Base Redetermination Date"})
training_data.append({"class":"Collateral", "sentence":"The repayment of the Indebtedness shall be secured by the following (the items and types of collateral described herein and/or in the Security Instruments being collectively referred to as the Collateral) pursuant to: (i) a mortgage lien (as applicable) encumbering not less than 80% of Borrower's proved producing and proved non-producing oil, gas and other leasehold and mineral interests (including, without limitation, behind-the pipe values), on a first priority basis, including without limitation, those properties situated in the State of Louisiana (collectively, the Mortgages), and (ii) a first priority security interest in substantially all of Borrower's personal property according to the terms of a certain Pledge, Security Agreement and Assignment instrument dated as of the Closing Date, in form and substance satisfactory to Bank"})
training_data.append({"class":"Additional Properties", "sentence":"Bank shall have the right to a first mortgage lien position on any and all hereafter acquired or owned producing oil and/or gas well(s) or properties of whatever type of Borrower that have been evaluated for purposes of determining the Collateral Borrowing Base, even though such well(s) or properties do not constitute Collateral or Proven Reserves as of the date of this Agreement, including, without limitation, all newly or hereafter acquired oil and/or gas wells or properties"})
training_data.append({"class":"Cross Default and Cross-Collateralization", "sentence":"It is the express intention and agreement of Borrower and Bank that any and all existing and future obligations, liabilities and indebtedness now or hereafter owing by Borrower to Bank (including the Revolver Note, and Letter of Credit Exposure and any Hedge Agreement) be and continuously remain cross-defaulted and cross-collateralized to the fullest extent permitted by applicable law with any and all other existing or future obligations, liabilities and indebtedness of Borrower to Bank or of Borrower to the Swap Counterparty"})
training_data.append({"class":"Keepwell", "sentence":"Each Qualified ECP Guarantor hereby jointly and severally absolutely, unconditionally and irrevocably undertakes to provide such funds or other support as may be needed from time to time by each other loan party to honor all of its obligations under guaranty instrument in respect of a Swap Obligation (provided, however, that each Qualified ECP Guarantor shall only be liable under this Section 3.4 for the maximum amount of such liability that can be hereby incurred without rendering its obligations under this Section 3.4 or otherwise under this Guaranty voidable under applicable law relating to fraudulent conveyance or fraudulent transfer, and not for any greater amount). Except as otherwise provided herein, the obligations of each Qualified ECP Guarantor under thisSection 3.4 shall remain in full force and effect until the termination of all Swap Obligations"})
training_data.append({"class":"Guaranty", "sentence":"To secure the prompt and full payment when due of the Indebtedness, the Borrower shall cause the Guarantor to execute and deliver to the Bank at Closing its Guaranty Agreement under which the Guarantor shall absolutely and unconditionally guaranty the prompt repayment of the Indebtedness"})
training_data.append({"class":"Redetermination of Collateral Borrowing Base", "sentence":"At any time within thirty (30) days of the receipt of such third party petroleum engineering report required by Section 4.1(a), and in no event later than each April 1 and October 1 (commencing April 1, 2019) (each being a Redetermination Date) Bank shall (i) make a good faith determination of the present worth using such pricing and discount factor (in no event shall the present worth be discounted by a factor less than nine percent (9.0%)) and advance rate as it deems appropriate pursuant to Bank's then applicable energy lending and engineering policies, procedures and pricing parameters, of the future net revenue estimated by Bank to be received by Borrower from not less than eighty percent (80%) of the oil and gas wells/properties so evaluated and attributable to Borrower, multiplied by a percentage then determined by Bank in good faith to be appropriate on the basis of Bank's then applicable energy lending criteria"})
training_data.append({"class":"Collateral Borrowing Base Deficiency", "sentence":"Should the sum of the unpaid outstanding principal balance of the Revolver Note at any time prior to maturity plus all other Indebtedness be greater than the Collateral Borrowing Base in effect at such time (a Deficiency), Bank may notify Borrower in writing of the deficiency.  Within fifteen (15) days from and after the date of any such deficiency notice Borrower shall notify Bank in writing"})
training_data.append({"class":"Conditions Precedent to Loan", "sentence":"The obligation of Bank to establish the Revolver Commitment and to make Revolver Loan advances, including the initial Loan advance hereunder, and to issue Letters of Credit, are subject to the satisfaction of all of the following conditions on or prior to the Closing Date (in addition to the other terms and conditions set forth herein)"})
training_data.append({"class":"No Default", "sentence":"There shall exist no Default or Event of Default on the Closing Date"})
training_data.append({"class":"Representations and Warranties", "sentence":"The representations, warranties and covenants set forth in Articles VI and VII shall be true and correct on and as of the Closing Date, with the same effect as though made on and as of the Closing Date"})
training_data.append({"class":"Borrower/Guarantor Certificates", "sentence":"Each of Borrower and Guarantor shall have delivered to Bank a Certificate, dated as of the Closing Date, and signed by the Member of Borrower and the President/Chief Executive Officer and Secretary of Guarantor in a manner in compliance with Borrower's Operating Agreement and Guarantor's Bylaws certifying (i) to the matters covered by the conditions specified in subsections (a) and (b) of thisSection 5.1, (ii) that each of Borrower and Guarantor has performed and complied with all agreements and conditions required to be performed or complied with by them in this Agreement prior to or on the Closing Date, (iii) to the name and signature of each member and/or officer, as applicable, authorized to execute and deliver the Loan Documents and any other documents, certificates or writings and to borrow under this Agreement, and (iv) to such other matters in connection with this Agreement which Bank shall determine to be advisable"})
training_data.append({"class":"Proceedings", "sentence":"On or before the Closing Date, all limited liability company proceedings of Borrower and all corporate proceedings of Guarantor shall be satisfactory in form and substance to Bank and its counsel; and Bank shall have received copies, in form and substance satisfactory to Bank and its counsel, of the Certificate/Articles of Organization/Formation and Operating Agreement of Borrower, and the Certificate of Incorporation and Bylaws of Guarantor, as adopted, authorizing the execution and delivery of the Loan Documents, the borrowings and, as applicable, guaranteeing, under this Agreement, and the granting of the security interests in the Collateral pursuant to the Security Instruments, to secure the payment of the Indebtedness"})
training_data.append({"class":"Loan Documents/Security Instruments", "sentence":"Borrower shall have delivered to Bank the Revolver Loan Agreement, and the Security Instruments, appropriately executed by all parties, witnessed and acknowledged to the satisfaction of Bank and dated as of the Closing Date, together with such financing statements, and other documents as shall be necessary and appropriate to perfect Bank's security interests in the Collateral covered by said Security Instruments"})
training_data.append({"class":"Mortgages", "sentence":"Borrower shall have executed and delivered the Mortgages to Bank in multiple recordable form counterparts as reasonably required by Bank"})
training_data.append({"class":"Guaranty Agreements", "sentence":"Borrower shall have caused the Guarantor to deliver its Guaranty Agreement to Bank, appropriately executed"})
training_data.append({"class":"ISDA Agreement", "sentence":"Borrower shall have executed and delivered any applicable ISDA Agreement to the Swap Counterparty, if any, in counterparts as reasonably required by the Swap Counterparty, within 15 days after the Closing Date"})
training_data.append({"class":"Intercreditor Agreement", "sentence":"Borrower shall have delivered any applicable Intercreditor Agreement to Bank in counterparts as reasonably required by Bank and the Swap Counterparty"})
training_data.append({"class":"Payoff/Lien Releases/UCC Terminations", "sentence":"Bank shall have received such other information, documents and assurances as shall be reasonably requested by Bank, including (i) as applicable, executed and recordable mortgage lien releases or recordable assignments thereof reasonably satisfactory to Bank, and UCC termination statements from any such existing lender regarding the Mortgaged Property or appropriate assignments thereto, and (ii) such other information with respect to the Mortgaged Property of Borrower as shall be reasonably requested by Bank"})
training_data.append({"class":"UCC Searches", "sentence":"Bank shall have a certified UCC search covering Borrower, as debtor, from the central filing office of such jurisdictions as Bank reasonably deems necessary or appropriate, and Bank shall receive such other information, certificates (including a current good standing certificates issued as to Borrower's entity status in such jurisdictions as may be required by applicable Law, resolutions, documents and assurances as Bank shall reasonably request"})
training_data.append({"class":"Closing Opinion", "sentence":"Borrower and Guarantor shall cause its legal counsel to provide to Bank a closing opinion addressed to Bank covering due organization, good standing, authorization, due execution, no violation of charter or organizational documents, and other matters customarily covered in such opinions for secured loan transactions of the size and type contemplated by this Agreement, in scope and substance reasonably acceptable to Bank"})
training_data.append({"class":"Conditions to All Extensions of Credit", "sentence":"The obligation of Bank to make any Revolver Loan or issue any letters of credit hereunder (including the initial Revolver Loan advance to be made hereunder) is subject to the satisfaction of the following additional conditions precedent on the date of making such Revolver Loan advance or issuing such letter of credit (in each case, in addition to the conditions set forth in Section 3.1 above, and in Article II)"})
training_data.append({"class":"Representations and Warranties", "sentence":"The representations and warranties made by Borrower herein and in any other Loan Document or which are contained in any certificate furnished at any time under or in connection herewith shall (i) on and as of the date of making the initial Loan advance, be true and correct and (ii) on and as of the date of making each other Revolver Loan advance or issuing a letter of credit, be true and correct in all material respects on as if made on and as of the date of such extension or such request, as applicable (except for those which expressly relate to an earlier specified date and except that any representations or warranties that already are qualified or modified as to materiality or material adverse effect in the text thereof, such representations and warranties shall be true and correct in all respects)"})
training_data.append({"class":"Event of Default", "sentence":"No Default or Event of Default shall have occurred and be continuing on such date or after giving effect to the Revolver Loan advance or Letter of Credit issuance to be made on such date and the application of the proceeds thereof unless such Default or Event of Default shall have been waived in accordance with this Agreement"})
training_data.append({"class":"Bankruptcy or Insolvency", "sentence":"No Bankruptcy Event shall have occurred by or with respect to Borrower or Guarantor"})
training_data.append({"class":"No Material Adverse Change", "sentence":"No circumstance, event or condition shall have occurred or be existing which would reasonably be expected to have a Material Adverse Change"})
training_data.append({"class":"Payment of Taxes and Claims", "sentence":"Borrower will pay and discharge or cause to be paid and discharged all Taxes imposed upon the income or profits of Borrower or upon the property, real, personal or mixed, or upon any part thereof, belonging to Borrower before the same shall be in default, and all lawful claims for labor, rentals, materials and supplies which, if unpaid, might become a Lien upon its property or any part thereof; provided however, that Borrower shall not be required to pay and discharge or cause to be paid or discharged any such Tax, assessment or claim so long as the validity thereof shall be contested in good faith by appropriate proceedings, and adequate book reserves shall be established with respect thereto, and Borrower shall pay such Tax, charge or claim before any property subject thereto shall become subject to execution"})
training_data.append({"class":"Maintenance of Legal Existence", "sentence":"Borrower will do or cause to be done all things necessary to preserve and keep in full force and effect its corporate existence, rights and franchises and will continue to conduct and operate its business substantially as being conducted and operated presently.  Borrower will become and remain qualified to conduct business in each jurisdiction where the nature of the business or ownership of property by Borrower may require such qualification"})
training_data.append({"class":"Preservation of Property", "sentence":"Borrower will at all times maintain, and take commercially reasonable steps to preserve and protect all franchises and trade names and keep all the remainder of its properties which are used or useful in the conduct of its businesses whether owned in fee or otherwise, in good repair and operating condition (ordinary wear and tear excepted). Borrower shall comply with all material leases to which it is a party or under which it occupies property so as to prevent any material loss or forfeiture thereunder"})
training_data.append({"class":"Insurance", "sentence":"To the extent customary in the oil and gas industry for similarly situated leasehold owners and producers, Borrower will keep or cause to be kept (whether Borrower or, if applicable, the operator of the Proven Reserves), adequately insured by financially sound and reputable insurers Borrower's property of a character usually insured by businesses engaged in the same or similar businesses, including the Collateral casualty/hazard insurance and business interruption insurance.  Upon written demand by Bank any insurance policies covering the Collateral shall be endorsed to provide for payment of losses to Bank as its interest may appear, to provide that such policies may not be cancelled, reduced or affected in any manner for any reason without prior notice to Bank, and to provide for any other matters which Bank may reasonably require.  Borrower shall at all times maintain or, where applicable, cause the operators of the Proven Reserves to maintain adequate insurance, by financially sound and reputable insurers, including without limitation, the following coverage's: [(i) insurance against damage to persons and property, including comprehensive general liability, worker's compensation and automobile liability, and (ii) insurance against sudden and accidental environmental and pollution hazards and accidents that may occur on the Mortgaged Property.  Borrower shall annually furnish to Bank reasonable evidence of its compliance with the requirements of this Section 6.4 within fifteen (15) days of renewal of the insurance required hereby"})
training_data.append({"class":"Quarterly Financial Statements", "sentence":"As soon as practicable after the end of every fiscal quarter of Borrower other than and except only for the fourth (4th) and final fiscal quarter of each fiscal year, and in any event within forty five (45) days thereafter, Borrower shall furnish to Bank the following internally prepared financial statements, on a sound accounting basis in accordance with GAAP, consistently applied"})
training_data.append({"class":"Compliance with Applicable Laws", "sentence":"Borrower will comply with the material requirements of all applicable Laws including with limitation, Occupational Safety and Health Administration (OSHAWA) provisions, rules, regulations and orders of any Tribunal and obtain any licenses, permits, franchises or other governmental authorizations necessary to the ownership of its properties or to the conduct of its business"})
training_data.append({"class":"Net Lease Operating Reports", "sentence":"No later than thirty (30) days after the end of each calendar quarter, reports regarding leases in the same form as they are received by the operator under each applicable operators agreement"})
training_data.append({"class":"Environmental Covenants", "sentence":"Except as commonly occurring in the normal and customary oil and gas exploration activities from time to time, Borrower will promptly notify Bank of and provide Bank with copies of any notifications of discharges or releases or threatened releases or discharges of a Polluting Substance on, upon, into or from the Collateral which are given or required to be given by or on behalf of Borrower to any federal, state or local Tribunal if any of the foregoing may materially and adversely affect Borrower or any part of the Collateral, and such copies of notifications shall be delivered to Bank at the same time as they are delivered to the Tribunal.  Borrower further agrees promptly to undertake and diligently pursue to completion any legally required remedial containment and cleanup action in the event of any release or discharge or threatened release or discharge of a Polluting Substance on, upon, into or from the Collateral"})
training_data.append({"class":"Hedge Reports", "sentence":"As soon as available on a quarterly basis and no later than the thirtieth (30th) day of each succeeding calendar quarter, the monthly trading statements, setting forth as of the last Business Day of such prior fiscal quarter end, a summary of its hedging positions, if any, under all Risk Management Agreements (including commodity price swap agreements, forward agreements or contracts of sale which provide for prepayment for deferred shipment or delivery of oil, gas or other commodities) of Borrower, identifying such matters as the type, term effective date, termination date and notional principal amounts or volumes, the hedged price(s), interest rate(s) or exchange rate(s), as applicable, and any new credit support agreements relating thereto not previously disclosed to Bank"})
training_data.append({"class":"Net Lease Operating Reports", "sentence":"No later than thirty (30) days after the end of each calendar quarter, reports regarding leases in the same form as they are received by the operator under each applicable operators agreement."})
training_data.append({"class":"Environmental Indemnities", "sentence":"Borrower hereby agrees to indemnify, defend and hold harmless Bank and each of its officers, directors, employees, agents, consultants, attorneys, contractors and each of its affiliates, successors or assigns, or transferees from and against, and reimburse said Persons in full with respect to, any and all loss, liability, damage, fines, penalties, costs and expenses, of every kind and character, including reasonable attorneys' fees and court costs, known or unknown, fixed or contingent, occasioned by or associated with any claims, demands, causes of action, suits and/or enforcement actions, including any administrative or judicial proceedings, and any remedial, removal or response actions ever asserted, threatened, instituted or requested by any Persons, including any Tribunal, arising out of"})
training_data.append({"class":"Notice of Default", "sentence":"Within five (5) Business Days after any officer becoming aware of any condition or event which constitutes an Event of Default or Default, Borrower will give Bank a written notice thereof specifying the nature and period of existence thereof and what actions, if any, Borrower is taking and proposes to take with respect thereto"})
training_data.append({"class":"Notice of Litigation", "sentence":"Within five (5) Business Days after becoming aware of the existence of any action, suit or proceeding at law or in equity before any Tribunal, an adverse outcome in which would (i) materially impair the ability of Borrower to carry on its businesses substantially as now conducted, (ii) materially and adversely affect the condition (financial or otherwise) of Borrower, or (iii) result in monetary damages in excess of $100,000, Borrower will give Bank a written notice specifying the nature thereof and what actions, if any, Borrower are taking and proposes to take with respect thereto"})
training_data.append({"class":"Notice of Claimed Default", "sentence":"Within five (5) Business Days after becoming aware that the holder of any note or any evidence of indebtedness or other security of Borrower has given notice or taken any action with respect to a claimed default or event of default thereunder, if the amount of the note or indebtedness exceeds $100,000, Borrower will give Bank a written notice specifying the notice given or action taken by such holder and the nature of the claimed default or event of default thereunder and what actions, if any, Borrower is taking and propose to take with respect thereto"})
training_data.append({"class":"Change of Management", "sentence":"Within five (5) Business Days after any change in the managers of Borrower or any officer of Borrower holding the office of President, Borrower shall give written notice thereof to Bank"})
training_data.append({"class":"Requested Information", "sentence":"With reasonable promptness, Borrower will give Bank such other data and information relating to Borrower's organization, financial results, and operations of the Collateral as from time to time may be reasonably requested by Bank"})
training_data.append({"class":"Inspection", "sentence":"Borrower will keep complete and accurate books and records with respect to the Collateral and its other properties, businesses and operations and upon reasonable advance notice will permit employees and representatives of Bank to review, audit, inspect and examine the same and to make copies thereof and extracts therefrom during normal business hours.  All such records (or accurate copies thereof if the original records are required by law, rule, regulation or ordinance to be kept in another location) shall be at all times kept and maintained at the offices of Borrower in Tulsa, Oklahoma.  Upon any Default or Event of Default, Borrower will surrender copies of all of such records relating to the Collateral to Bank upon receipt of any request therefor from Bank.  Borrower shall promptly notify Bank of any change in the location of its principal office"})
print("%s sentences in training data" % len(training_data))

words = []
classes = []
documents = []
ignore_words = ['?','(',')',',']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)

training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w])
print (training[i])
print (output[i])

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)
    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1
    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
    for j in iter(range(epochs+1)):
        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        # how much did we miss the target value?
        layer_2_error = y - layer_2
        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update
    now = datetime.datetime.now()
    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),'datetime': now.strftime("%Y-%m-%d %H:%M"),'words': words,'classes': classes}
    synapse_file = "synapses.json"
    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)

X = np.array(training)
y = np.array(output)
start_time = time.time()
train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)
elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")

# probability threshold
ERROR_THRESHOLD = 0.5
# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print("%s \n classification: %s" % (sentence, return_results))
    return return_results

with open("C:/Users/rroutr01/Desktop/contract.txt", "r") as ins:
    for line in ins:
        if (line != '\n'):
            result = classify(line)