



class NormFunctions:
    """
    This class provides normalization functions for different types of data.

    Attributes:
        - None
    """

    def __init__(self, arguments):
        self.params = arguments
    
    def _setnormcolumnsinrxn(self, rxn, normalizer, column):
        """
        This function normalizes a specified column or list of columns in a reaction dataframe (rxn)
        using a provided normalizer. The normalized values are stored in new columns with a "_normed" suffix.

        Parameters:
        - rxn: The reaction dataframe containing the data.
        - normalizer: The normalizer object used for normalization. If None, no normalization is applied.
        - column: The column(s) to be normalized. Can be a single column or a list of columns.

        Logic:
        1. Check if a normalizer is provided:
            a. Apply the normalizer's transform method to the specified column(s) in the dataframe.
        2. Check if `column` is a list or list-like object:
            a. Iterate through each column in the list.
            b. Store the normalized values in new columns with a "_normed" suffix.
        3. If `column` is a single column:
            a. Store the normalized value in a new column with a "_normed" suffix.
        4. Return the updated reaction dataframe.

        Returns:
        - rxn: The updated reaction dataframe with normalized columns.
        """
        if normalizer is not None:
            res = normalizer.transform(rxn[column])  # Apply normalization if a normalizer is provided
        if isinstance(column, list):  # Check if the column is a list or list-like
            for i, outputs in enumerate(column):  # Iterate through the list of columns
                rxn[f'{outputs}_normed'] = res[i]  # Store normalized values in new columns
        else:
            rxn[f'{column}_normed'] = res  # Store normalized value in a new column for a single column
        return rxn  # Return the updated reaction dataframe
    

    def _setnormstoinfo(self, rxn, column, normalizer):
        """
        This function updates the `info` dictionary with normalized or original values from a reaction dataframe (`rxn`).

        Parameters:
        - rxn: The reaction dataframe containing the data.
        - column: The column(s) to be updated in the `info` dictionary. Can be a single column or a list of columns.
        - normalizer: The normalizer object used for normalization. If None, original values are used.

        Logic:
        1. Check if `column` is a list or list-like object:
            a. Iterate through each column in the list.
            b. If a normalizer is provided, update `info` with the normalized values from `rxn`.
            c. If no normalizer is provided, update `info` with the original values from `rxn`.
        2. If `column` is a single column:
            a. If a normalizer is provided, update `info` with the normalized value from `rxn`.
            b. If no normalizer is provided, update `info` with the original value from `rxn`.
        """
        if isinstance(column, list):
            for outputs in column:
                if normalizer is not None:
                    self.info[outputs] = rxn[f'{outputs}_normed']
                else:
                    self.info[outputs] = rxn[outputs]
        else:
            if normalizer is not None:
                self.info[column] = rxn[f'{column}_normed']
            else:
                self.info[column] = rxn[column]

    def AddTargets(self,rxn):
        if hasattr(self,'target_normalizer'):
            self._setnormstoinfo(rxn,self.params.target,self.target_normalizer)
        else:
            self.target_normalizer = None
        self._setnormstoinfo(rxn,self.params.target,self.target_normalizer)

    def AddAdditionals(self,rxn):
        if self.params.additional is not None:
            if hasattr(self,'additional_normalizer'):
                self._setnormstoinfo(rxn,self.params.target,self.target_normalizer)
            else:
                self.target_normalizer = None
            self._setnormstoinfo(rxn,self.params.additional,self.params.additional_normalizer)